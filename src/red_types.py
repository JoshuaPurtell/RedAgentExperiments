from typing import List, Any, Tuple
from matplotlib.transforms import BlendedGenericTransform
import numpy as np
from pathlib import Path
import mediapy as media
import hnswlib
import time
from src.globals import output_full
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
import nltk
#nltk.download('brown')
from nltk.corpus import brown

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
    hps: List[int]
    max_hps: List[int]

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
    center_of_mass: Tuple[float, float]
    raw_texts_seen: List[Tuple[str, int]] #text, step
    texts_seen: List[Tuple[str, int, float]] #text, step, distance
    text_history_handler: Any

class VideoHandlerNew:
    def __init__(self, save_path, reset_count, instance_id):
        self.base_directory = Path(save_path) / 'rollouts'
        self.base_directory.mkdir(exist_ok=True)
        self.full_video_name = self.base_directory / f'full_reset_{reset_count}_id{instance_id}.mp4'
        self.model_video_name = self.base_directory / f'model_reset_{reset_count}_id{instance_id}.mp4'

        self.full_frame_writer = None
        self.model_frame_writer = None

    def open(self):
        self.full_frame_writer = media.VideoWriter(str(self.full_video_name), (128, 40), fps=60)
        self.model_frame_writer = media.VideoWriter(str(self.model_video_name), (128, 40), fps=60)

    def close(self):
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
        self.cached_median = None
        self.sorted_sample_cache = None
        self.sample_length = None
        self.cache_refresh_rate = 0.05

    def _add_distances_to_cache(self, selected_elements):
        selected_elements = np.array(selected_elements)  # Convert to NumPy array if not already
        dist_matrix = np.sqrt(np.sum((selected_elements[:, np.newaxis, :] - selected_elements[np.newaxis, :, :]) ** 2, axis=-1))
        for i in range(len(selected_elements)):
            for j in range(i + 1, len(selected_elements)):
                self.cached_distances[(i, j)] = dist_matrix[i][j]

    def get_min_distance_distribution(self, knn_index, number_of_samples=300):
        current_count = knn_index.get_current_count()
        if np.random.rand() < 1/np.sqrt(len(self.cached_distances)+1) or len(self.cached_distances) < 300:
            indices = np.arange(current_count)
            if current_count < number_of_samples:
                random_indices = indices
            else:
                random_indices = np.random.choice(indices, size=number_of_samples, replace=False)
            selected_vectors = knn_index.get_items(random_indices)
            self._add_distances_to_cache(selected_vectors)
        return list(self.cached_distances.values()) if self.cached_distances else [0]

    def update_frame_knn_index(self, flate_state, lmbda=2):
        current_count = self.knn_index.get_current_count()
        if current_count == 0:
            distances = [[0]]
            self.knn_index.add_items(flate_state, np.array([current_count]))
        else:
            _, distances = self.knn_index.knn_query(flate_state, k=1)
            if distances[0][0] > self.similar_frame_distance:
                self.knn_index.add_items(flate_state, np.array([current_count]))

        sample = self.get_min_distance_distribution(self.knn_index, number_of_samples=300)
        if len(sample) == 0:
            return 1

        if self.sorted_sample_cache is None or np.random.rand() < self.cache_refresh_rate:
            self.sorted_sample_cache = sorted(sample)
            self.sample_length = len(sample)

        reverse_quantile = 1 - np.searchsorted(self.sorted_sample_cache, distances[0][0]) / self.sample_length
        if not sample or set(sample) == {0} or distances[0][0] == 0:
            return 1 + lmbda * reverse_quantile
        else:
            if self.cached_median is None or np.random.rand() < self.cache_refresh_rate:
                self.cached_median = np.median(self.sorted_sample_cache)
                if self.cached_median == 0:
                    self.cached_median = max(self.sorted_sample_cache, default=1)
            self.cache_refresh_rate = 0.05 / np.sqrt(len(self.cached_distances) + 1)
            return self.log_like(distances[0][0] - self.cached_median + 2.8) + lmbda * reverse_quantile

    @staticmethod
    def log_like(x):
        if x > 1:
            return np.log(x)
        elif x < -1:
            return -np.log(-x)
        else:
            return x
class VisualHistoryKNNJosh:
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
            tquery = time.time()
            _, distances = self.knn_index.knn_query(flate_state, k = 1)
            if distances[0][0] > self.similar_frame_distance:
                self.knn_index.add_items(
                    flate_state, np.array([self.knn_index.get_current_count()])
                )
            tqeury_end = time.time()
            #print(f"query time: {tqeury_end-tquery}")
        tget_min_dist = time.time()
        sample = self.get_min_distance_distribution(self.knn_index, number_of_samples = 300)
        tget_min_dist_end = time.time()
        ##print(f"get min dist time: {tget_min_dist_end-tget_min_dist}")
        reverse_quantile = 1 - np.searchsorted(sorted(sample), distances[0][0]) / len(sample)
        if len(sample) == 0 or list(set(sample)) == [0]:
            return 1 + lmbda*reverse_quantile
        elif distances[0][0] == 0:
            return 1 + lmbda*reverse_quantile
        else:
            def log_like(x):
                if x > 1:
                    return np.log(x)
                elif x < -1:
                    return -np.log(-x)
                else:
                    return x
            return log_like(distances[0][0] / np.median(sample)+2.8) + lmbda*reverse_quantile

class TextHistoryHandler:
    #only use text if it includes prev text and is not included by following text
    #recent
    def __init__(self):
        self.tvs = TextVectorStore(similarity_threshold=0.1)
    
    def is_penult_fully_revealed(self, texts: List[Tuple[str, int]], recent = 5):
        if len(texts) < recent:
            return False
        recent_texts = texts[-recent:]
        if (recent - sum([0==len(t[0]) for t in recent_texts]))< 2:
            return False
        if len([1.0 == d for d in np.diff([t[1] for t in recent_texts])]) < 1: #have there been sequential text reveals?
            return False
        penult_and_pre_text_intersections = [list(set(t[0]).intersection(set(recent_texts[-1][0]))) for t in recent_texts[:-2]]
        texts_included_in_penult = sum([len(intersection) / len(list(set(text[0]))) > 0.8 for intersection, text in zip(penult_and_pre_text_intersections, recent_texts[:-2])])
        penult_inclusion_ratio = len(list(set(recent_texts[-2][0]).intersection(set(recent_texts[-1][0])))) / (len(list(set(recent_texts[-2][0])))+1)
        penult_strictly_included_in_last = (penult_inclusion_ratio > 0.3) and (penult_inclusion_ratio < 0.85)
        penult_strictly_equal_to_last = penult_inclusion_ratio > 0.97
        if texts_included_in_penult > 1 and not penult_strictly_included_in_last and not penult_strictly_equal_to_last:
            return True
        return False

    def get_final_text(self, texts: List[Tuple[str, int]]):
        if len(texts) < 2:
            return ""
        if self.is_penult_fully_revealed(texts):
            return texts[-2][0]
        return ""
    
    def get_final_text_distance(self, text: str):
        distance = self.tvs.get_distance_and_update_store(text)
        return distance

    
class TextVectorStore:
    def __init__(self, similarity_threshold=0.1):
        
        self.text_vectorizer = TfidfVectorizer()
        self.vector_dimension = None  # Initialize without setting dimension
        self.similarity_threshold = similarity_threshold
        self.vector_index = None  # Initialize as None

        corpus = [brown.raw(fileid) for fileid in brown.fileids()]
        self.fit_vectorizer(corpus)

    def _convert_text_to_vector(self, text):
        return self.text_vectorizer.transform([text]).toarray()[0]

    def _add_text_vector_to_store(self, text_vector):
        text_vector = np.array([text_vector], dtype='float32')  # Ensure text_vector is 2D and of type float32
        faiss.normalize_L2(text_vector)
        self.vector_index.add(text_vector.reshape(1, -1))  # Reshape the text_vector to 2D before adding


    def fit_vectorizer(self, corpus):
        self.text_vectorizer.fit(corpus)
        self.vector_dimension = len(self.text_vectorizer.get_feature_names_out())  # Set the correct dimension
        self.vector_index = faiss.IndexFlatL2(self.vector_dimension)  # Reinitialize the FAISS index

    def get_distance_and_update_store(self, text):
        text_vector = self._convert_text_to_vector(text)
        text_vector = np.array([text_vector])  # Ensure text_vector is 2D
        distances, _ = self.vector_index.search(text_vector, 1)
        if distances[0][0] > self.similarity_threshold:
            self._add_text_vector_to_store(text_vector)
        return distances[0][0]
