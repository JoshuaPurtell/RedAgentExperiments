from typing import List, Any
from matplotlib.transforms import BlendedGenericTransform
import numpy as np
from pathlib import Path
import mediapy as media
import hnswlib
import time
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

class VideoHandlerNew:
    def __init__(self, save_path, reset_count, instance_id):
        self.base_directory = Path(save_path) / 'rollouts'
        self.base_directory.mkdir(exist_ok=True)
        self.full_video_name = self.base_directory / f'full_reset_{reset_count}_id{instance_id}.mp4'
        self.model_video_name = self.base_directory / f'model_reset_{reset_count}_id{instance_id}.mp4'

        self.full_frame_writer = None
        self.model_frame_writer = None

    def open(self):
        # Open video writers
        self.full_frame_writer = media.VideoWriter(str(self.full_video_name), (128, 40), fps=60)
        self.model_frame_writer = media.VideoWriter(str(self.model_video_name), (128, 40), fps=60)

    def close(self):
        # Close video writers
        if self.full_frame_writer:
            self.full_frame_writer.close()
        if self.model_frame_writer:
            self.model_frame_writer.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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

class VisualHistoryKNN:
    def __init__(self, vector_dimension, number_of_elements, similar_frame_distance):
        self.vector_dimension = vector_dimension
        self.number_of_elements = number_of_elements
        self.similar_frame_distance = similar_frame_distance
        self.knn_index = hnswlib.Index(space='l2', dim=self.vector_dimension)
        self.knn_index.init_index(max_elements=self.number_of_elements, ef_construction=100, M=16)
        self.cached_distances = {}

    def _add_distances_to_cache(self, selected_elements: List[np.ndarray]):
        for i in range(len(selected_elements)):
            for j in range(len(selected_elements)):
                if i != j:
                    pair = tuple(sorted([i, j]))
                    if pair not in self.cached_distances:
                        distance = np.linalg.norm(selected_elements[i]-selected_elements[j]) 
                        self.cached_distances[pair] = abs(distance)

    def get_min_distance_distribution(self, knn_index, number_of_samples = 300):
        if np.random.rand() < 0.05 or len(self.cached_distances) < 300:
            indices = np.arange(knn_index.get_current_count())
            all_vectors = np.array(knn_index.get_items(indices))
            if len(all_vectors) < number_of_samples:
                random_indices = np.arange(len(all_vectors))
            else:
                random_indices = np.random.choice(len(all_vectors), size = min(number_of_samples, len(all_vectors)), replace=False)
            selected_vectors = all_vectors[random_indices]
            self._add_distances_to_cache(selected_vectors)
        minimum_distances = list(self.cached_distances.values())
        return minimum_distances if len(minimum_distances) > 0 else [0]
    
    def update_frame_knn_index(self, flate_state, lmbda = 2):
        if self.knn_index.get_current_count() == 0:
            distances = [[0]]
            self.knn_index.add_items(
                flate_state, np.array([self.knn_index.get_current_count()])
            )
        else:
            _, distances = self.knn_index.knn_query(flate_state, k = 1)
            if distances[0][0] > self.similar_frame_distance:
                self.knn_index.add_items(
                    flate_state, np.array([self.knn_index.get_current_count()])
                )
        sample = self.get_min_distance_distribution(self.knn_index, number_of_samples = 300)
        reverse_quantile = 1 - np.searchsorted(sorted(sample), distances[0][0]) / len(sample)
        if len(sample) == 0 or list(set(sample)) == [0]:
            return 1 + lmbda*reverse_quantile
        elif distances[0][0] == 0:
            return 1 + lmbda*reverse_quantile
        else:
            return np.log(distances[0][0] / np.median(sample)+2.8) + lmbda*reverse_quantile
    