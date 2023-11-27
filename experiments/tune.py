from typing import Dict, Any
import numpy as np

from experiments.hyperparam_tuning import Tuner

ep_length = 2048* 10#204, 2048* 10
batch_size = 64
n_epochs = 1

def value_function(env_infos: Dict[str,Any]) -> float:
    badges = np.mean([env_info['badges'] for env_info in env_infos])
    avg_levels = np.mean([np.mean([level for level in env_info['levels'] if level > 0]) for env_info in env_infos])
    seen_pokemon = np.mean([len(env_info['seen_pokemon']) for env_info in env_infos])
    op_level = np.mean([env_info['op_level'] for env_info in env_infos])
    return badges + .01 * avg_levels + .05 * seen_pokemon + .03 * op_level

env_config_tuning = {
    "headless": True,
    "save_final_state": True,
    "early_stop": False,
    "action_freq": 24,
    "init_state": "storage/has_pokedex_nballs_copy.state",
    "max_steps": ep_length,
    "print_rewards": True,
    "save_video": True,
    "fast_video": True,
    "gb_path": "storage/PokemonRed.gb",
    "debug": False,
    "sim_frame_dist": 200_000.0,
    "use_screen_explore": True,
    "reward_scale": 4,
    "extra_buttons": False,
    "explore_weight": 5,  # 2.5
}

hyperparameter_config = {
    "badge_reward": 100.0,
    "seen_pokemon_reward": 1.0,
    "op_level_reward": 1.0,
    "p_types_reward": 1.0,
    "rel_number_of_times_weve_been_here": 1.0,
    "number_of_spots": 1.0,
    "distance_from_center": 1.0,
    "novelty": 1.0,
    "heal_reward": 1.0,
    "money_reward": 1.0,
    "died_reward": 1.0,
    "fainted_reward": 1.0,
    "story_weight": 0.25,
    "experience_weight": 0.25,
    "exploration_weight": 0.25,
    "tactics_weight": 0.25,
    "text_weight": 0.05
}

if __name__ == '__main__':
    tuner = Tuner(env_config_tuning, ep_length, batch_size, n_epochs, hyperparameter_config, value_function)
    score = tuner.execute_run()
    print("Got this score: ", score)