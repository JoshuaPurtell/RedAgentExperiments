from typing import List, Any
import numpy as np
from pathlib import Path
import mediapy as media

from src.globals import output_full

class PlayerState:
    step: int
    x_pos: int
    y_pos: int
    map_n: int
    map_location: str
    last_action: int
    pcount: int
    op_level: int
    levels: List[int]
    ptypes: List[int]
    money: int
    badges: int
    seen_pokemon: List[int]
    hp_fracs: List[float]

class Reward:
    story_reward: float
    experience_reward: float
    exploration_reward: float
    tactics_reward: float

    total_reward: float
    channel_vec: np.ndarray

    def null(self):
        self.story_reward = 0
        self.experience_reward = 0
        self.exploration_reward = 0
        self.tactics_reward = 0
        self.total_reward = 0
        self.channel_vec = np.array([0,0,0,0])

#move toward H = List[h]
class History:
    rewards = List[Reward]
    agent_states: List[PlayerState]
    recent_frames: list
    recent_memory: list
    recent_actions: list
    seen_coords: dict


class VideoHandler:
    base_directory: str
    full_video_name: str
    model_video_name: str
    full_frame_writer: Any
    model_frame_writer: Any

    def __init__(self, save_path, reset_count, instance_id):
        self.base_directory = save_path / Path('rollouts')
        self.base_directory.mkdir(exist_ok=True)
        self.full_video_name = Path(f'full_reset_{reset_count}_id{instance_id}').with_suffix('.mp4')
        self.model_video_name = Path(f'model_reset_{reset_count}_id{instance_id}').with_suffix('.mp4')
        self.full_frame_writer = media.VideoWriter(self.base_directory / self.full_video_name, (128, 40), fps=60)
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(self.base_directory / self.model_video_name, (128, 40), fps=60)
        self.model_frame_writer.__enter__()