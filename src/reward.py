import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from src.game import GameHandler
from src.red_types import History, Reward, VisualHistoryKNN


class RewardHandler:
    # story / strategy
    badge_reward: float
    seen_pokemon_reward: float

    # experience
    op_level_reward: float
    p_types_reward: float

    # explore
    seen_coords_relative_len: float

    # tactics
    heal_reward: float
    money_reward: float

    reward_scale: float
    explore_weight: float

    hyperparameters: Dict[str, Any]

    def __init__(self, reward_scale=1, explore_weight=1, reward_hyperparameters: Dict[str, Any] = {}):
        self.reward_scale = reward_scale
        self.explore_weight = explore_weight
        self.hyperparameters = reward_hyperparameters

    def get_story_reward(self, history: History):
        self.badge_reward = history.agent_states[-1].badges - history.agent_states[-2].badges
        self.seen_pokemon_reward = sum(history.agent_states[-1].seen_pokemon) - sum(
            history.agent_states[-2].seen_pokemon
        )
        story_param_vec = [self.hyperparameters["badge_reward"], self.hyperparameters["seen_pokemon_reward"]]
        reward_vec = [self.badge_reward, self.seen_pokemon_reward]
        return np.dot(story_param_vec, reward_vec)

    def get_experience_reward(self, history: History):
        self.op_level_reward = history.agent_states[-1].op_level - np.max(
            [history.agent_states[i].op_level for i in range(0, len(history.agent_states) - 1)]
        )
        self.p_types_reward = len(set(history.agent_states[-1].ptypes)) - len(
            set(history.agent_states[-2].ptypes)
        )  # ?
        self.levels_reward = sum(history.agent_states[-1].levels) - sum(history.agent_states[-2].levels)
        exp_param_vec = [
            self.hyperparameters["op_level_reward"],
            self.hyperparameters["p_types_reward"],
            self.hyperparameters["levels_reward"],
        ]
        reward_vec = [self.op_level_reward, self.p_types_reward, self.levels_reward]
        return np.dot(exp_param_vec, reward_vec)

    def get_visual_novelty_reward(self, flat_state, visual_history_knn: VisualHistoryKNN):
        reward = visual_history_knn.update_frame_knn_index(flat_state)
        return reward, visual_history_knn

    def get_text_reward(self, history: History):
        if len(history.texts_seen) != 0 and history.texts_seen[-1][1] == len(history.agent_states) - 2:
            return np.log(history.texts_seen[-1][2] + 1)
        else:
            return 0

    def get_exploration_reward(
        self, history: History, gamehandler: GameHandler, visual_history_knn: VisualHistoryKNN, state
    ):
        position = gamehandler.get_pos()
        x_pos = position["x"]
        y_pos = position["y"]
        map_n = position["map_n"]
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        rel_number_of_times_weve_been_here = np.max([0,np.log(len(history.seen_coords[coord_string])+2) - np.median(
            [np.log(len(v)+2) for v in history.seen_coords.values()]
        )])
        number_of_spots = np.log(len(history.seen_coords))
        distance_from_center = np.sqrt(np.linalg.norm(np.array([x_pos, y_pos]) - history.center_of_mass))

        visual_novelty, visual_history_knn = self.get_visual_novelty_reward(state, visual_history_knn)
        # exploration_param_vec = [-.5,.5,3,2]
        exploration_param_vec = [
            self.hyperparameters["rel_number_of_times_weve_been_here"],
            self.hyperparameters["number_of_spots"],
            self.hyperparameters["distance_from_center"],
            self.hyperparameters["visual_novelty"],
        ]
        reward_vec = [rel_number_of_times_weve_been_here, number_of_spots, distance_from_center, visual_novelty]
        return np.dot(exploration_param_vec, reward_vec), visual_history_knn

    def get_tactics_reward(self, history: History, gamehandler: GameHandler):
        self.fainted_reward = sum([hp == 0 for hp in history.agent_states[-1].hps]) - sum(
            [hp == 0 for hp in history.agent_states[-1].max_hps]
        )
        self.heal_reward = sum(history.agent_states[-1].hps) - sum(history.agent_states[-2].hps)
        self.money_reward = history.agent_states[-1].money - history.agent_states[-2].money
        died = gamehandler.get_death()
        # tactics_param_vec = [.5,.5, -10, -3]
        tactics_param_vec = [
            self.hyperparameters["heal_reward"],
            self.hyperparameters["money_reward"],
            self.hyperparameters["died_reward"],
            self.hyperparameters["fainted_reward"],
        ]
        reward_vec = [self.heal_reward, self.money_reward, 1 if died else 0, self.fainted_reward]
        return np.dot(tactics_param_vec, reward_vec)

    def compute_reward(self, history: History, gamehandler: GameHandler, visual_history_knn: VisualHistoryKNN, state):
        # text_weight = .05
        # channel_params = [.25,.25,self.explore_weight*.25,.25, text_weight*.25]
        # t0 = time.time()
        channel_params = [
            self.hyperparameters["story_weight"],
            self.hyperparameters["experience_weight"],
            self.explore_weight * self.hyperparameters["exploration_weight"],
            self.hyperparameters["tactics_weight"],
            self.hyperparameters["text_weight"],
        ]
        if len(history.agent_states) < 2:
            reward = Reward()
            reward.story_reward = 0
            reward.experience_reward = 0
            reward.exploration_reward = 0
            reward.tactics_reward = 0

            reward.channel_vec = [0] * len(channel_params)
            reward.total_reward = 0
            return reward, visual_history_knn

        story_reward = self.get_story_reward(history)
        experience_reward = self.get_experience_reward(history)
        exploration_reward, visual_history_knn = self.get_exploration_reward(
            history, gamehandler, visual_history_knn, state
        )
        tactics_reward = self.get_tactics_reward(history, gamehandler)
        text_reward = self.get_text_reward(history)

        channel_vec = [story_reward, experience_reward, exploration_reward, tactics_reward, text_reward]
        total_reward = self.reward_scale * np.dot(channel_params, channel_vec)

        reward = Reward()
        reward.story_reward = story_reward
        reward.experience_reward = experience_reward
        reward.exploration_reward = exploration_reward
        reward.tactics_reward = tactics_reward
        if story_reward > 0 or experience_reward > 0:
            print(
                f"story_reward: {story_reward * channel_params[0]}, experience_reward: {experience_reward * channel_params[1]}, exploration_reward: {exploration_reward * channel_params[2]}, tactics_reward: {tactics_reward * channel_params[3]}, text_reward: {text_reward * channel_params[4]}, total_reward: {total_reward}"
            )
        reward.channel_vec = channel_vec
        reward.total_reward = total_reward
        # t1 = time.time()
        # print(f"reward took {t1-t0} seconds")
        return reward, visual_history_knn
