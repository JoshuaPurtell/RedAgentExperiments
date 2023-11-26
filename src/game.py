import re
from typing import List

import numpy as np

from src.device import DeviceHandler
from src.globals import frame_stacks, memory_height, output_shape
from src.red_types import History, PlayerState, Reward, TextHistoryHandler
from src.text import dump_text, get_text


class GameHandler:
    def __init__(self, devicehandler: DeviceHandler):
        self.history = History()
        self.history.center_of_mass = (0, 0)
        self.history.rewards = []
        self.history.texts_seen = []
        self.history.raw_texts_seen = []
        self.history.agent_states = []
        self.history.recent_frames = np.zeros(
            (frame_stacks, output_shape[0], output_shape[1], output_shape[2]), dtype=np.uint8
        )
        self.history.recent_memory = np.zeros((output_shape[1] * memory_height, 3), dtype=np.uint8)
        self.history.recent_actions = []
        self.history.seen_coords = {}
        self.history.text_history_handler = TextHistoryHandler()
        self.devicehandler = devicehandler

    def get_levels(self):
        return [max(self.devicehandler.read_m(a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]

    def get_party_types(self):
        return [self.devicehandler.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]

    def read_bcd(self, num: int) -> int:
        return 10 * ((num >> 4) & 0x0F) + (num & 0x0F)

    def get_money(self) -> int:
        return (
            100 * 100 * self.read_bcd(self.devicehandler.read_m(0xD347))
            + 100 * self.read_bcd(self.devicehandler.read_m(0xD348))
            + self.read_bcd(self.devicehandler.read_m(0xD349))
        )

    def read_triple(self, start_add: int) -> int:
        return (
            256 * 256 * self.devicehandler.read_m(start_add)
            + 256 * self.devicehandler.read_m(start_add + 1)
            + self.devicehandler.read_m(start_add + 2)
        )

    def get_exps(self) -> List[int]:
        poke_xps = [self.read_triple(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        return poke_xps

    def get_seen_pokemon(self) -> List[int]:
        return [self.devicehandler.bit_count(self.devicehandler.read_m(i)) for i in range(0xD30A, 0xD31D)]

    def get_badges(self) -> int:
        return self.devicehandler.bit_count(self.devicehandler.read_m(0xD356))

    def get_hps(self) -> List[int]:
        hps = [self.devicehandler.read_m(addr) for addr in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]]
        max_hps = [self.devicehandler.read_m(addr) for addr in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]]
        return hps, max_hps

    def get_n_events(self) -> int:
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        museum_ticket = (0xD754, 0)
        base_event_flags = 13
        return max(
            sum(
                [
                    self.devicehandler.bit_count(self.devicehandler.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags
            - int(self.devicehandler.read_bit(museum_ticket[0], museum_ticket[1])),
            0,
        )

    def get_pos(self) -> dict:
        return {
            "x": self.devicehandler.read_m(0xD362),
            "y": self.devicehandler.read_m(0xD361),
            "map_n": self.devicehandler.read_m(0xD35E),
        }

    def get_opponent_level(self) -> int:
        opponent_level = (
            max([self.devicehandler.read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]) - 5
        )
        return opponent_level

    def get_death(self):
        if sum(self.get_hps()[0]) == 0 and sum(self.get_hps()[1]) > 0:
            return True

    def screen_to_text(self, tile_size=8, mapping_dict=None):
        dump_text(self.devicehandler.pyboy, tile_hash_map_path="run/hash_to_english.json")
        return get_text(self.devicehandler.pyboy)
        # fill in later

    def update_recent_memory(self, new_observation: np.ndarray):
        self.history.recent_memory = np.roll(self.history.recent_memory, 3)
        self.history.recent_memory[0, 0] = min(new_observation[0] * 64, 255)
        self.history.recent_memory[0, 1] = min(new_observation[1] * 64, 255)
        self.history.recent_memory[0, 2] = min(new_observation[2] * 128, 255)

    def update_rewards(self, reward: Reward):
        self.history.rewards.append(reward)

    def update_raw_texts(self, step_count: int):
        current_text = self.screen_to_text()
        if set(current_text).difference(set(["X"," ","","\n"])):
            current_text = re.sub('X{2,}', '', current_text)
            current_text = re.sub('X\s{1,}X', '', current_text)
            current_text = re.sub('\s{2,}', ' ', current_text)
            self.history.raw_texts_seen.append((current_text, step_count))
    
    def update_texts(self, step_count: int):
        penult_text_final = self.history.text_history_handler.get_final_text(self.history.raw_texts_seen[-10:])
        if len(penult_text_final) > 0:
            distance_from_seen_texts = self.history.text_history_handler.get_final_text_distance(penult_text_final)
            self.history.texts_seen.append((penult_text_final, step_count-1, distance_from_seen_texts))

    def update_seen_coords(self, step_count: int):
        position = self.get_pos()
        x_pos = position["x"]
        y_pos = position["y"]
        map_n = position["map_n"]
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"

        def extract_coords(coord_string):
            x_part = re.search("x:(.*) y:", coord_string).group(1)
            y_part = re.search("y:(.*) m:", coord_string).group(1)
            return int(x_part), int(y_part)
        
        if coord_string not in self.history.seen_coords:
            self.history.seen_coords[coord_string] = [step_count]
            x_pos, y_pos = extract_coords(coord_string)
            self.history.center_of_mass = np.mean(
                [np.array(extract_coords(coord_string)) for coord_string in self.history.seen_coords.keys()], axis=0
            )
        else:
            self.history.seen_coords[coord_string].append(step_count)

    def update_agent_states(self, action):
        new_state = PlayerState()
        position = self.get_pos()
        new_state.x_pos = position["x"]
        new_state.y_pos = position["y"]
        new_state.map_n = position["map_n"]
        new_state.step = len(self.history.agent_states)
        new_state.map_location = self.devicehandler.read_m(0xD35F)
        new_state.last_action = action
        new_state.seen_pokemon = self.get_seen_pokemon()
        new_state.op_level = self.get_opponent_level()
        new_state.levels = self.get_levels()
        new_state.ptypes = self.get_party_types()
        hps, max_hps = self.get_hps()
        new_state.hps = hps#[hps[i] / np.max([1, max_hps[i]]) for i in range(6)]
        new_state.max_hps = max_hps
        new_state.badges = self.get_badges()
        new_state.money = self.get_money()

        self.history.agent_states.append(new_state)
