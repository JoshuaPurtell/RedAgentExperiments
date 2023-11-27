from typing import Any, Dict
import numpy as np
import optuna
import json

from experiments.hyperparam_tuning import Tuner

ep_length = 2048 * 5  # 204, 2048* 10
batch_size = 32
n_epochs = 1


def value_function(env_infos: Dict[str, Any]) -> float:
    badges = np.mean([env_info["badges"] for env_info in env_infos])
    avg_levels = np.mean([np.mean([level for level in env_info["levels"] if level > 0]) for env_info in env_infos])
    seen_pokemon = np.mean([len(env_info["seen_pokemon"]) for env_info in env_infos])
    op_level = np.mean([env_info["op_level"] for env_info in env_infos])
    return 10*badges + 0.01 * avg_levels + 0.05 * seen_pokemon + 0.03 * op_level


env_config_tuning = {
    "headless": True,
    "save_final_state": True,
    "early_stop": False,
    "action_freq": 24,
    "init_state": "storage/has_pokedex_nballs_copy.state",
    "max_steps": ep_length,
    "print_rewards": False,
    "save_video": False,
    "fast_video": True,
    "gb_path": "storage/PokemonRed.gb",
    "debug": False,
    "sim_frame_dist": 200_000.0,
    "use_screen_explore": True,
    "reward_scale": 1,
    "extra_buttons": False,
    "explore_weight": 5,  # 2.5
}

class ParameterOptimizer:
    def __init__(self, n_trials: int = 30):
        self.n_trials = n_trials

    def objective(self, trial: optuna.Trial) -> float:
        hyperparameter_config = {
            "badge_reward": trial.suggest_float("badge_reward", 500.0, 1500.0),
            "seen_pokemon_reward": trial.suggest_float("seen_pokemon_reward", 1.0, 3.0),
            "op_level_reward": trial.suggest_float("op_level_reward", 0.1, 1.5),
            "levels_reward": trial.suggest_float("levels_reward", 0.1, 1.5),
            "p_types_reward": trial.suggest_float("p_types_reward", 0.1, 1.5),
            "rel_number_of_times_weve_been_here": trial.suggest_float("rel_number_of_times_weve_been_here", -1.0, 0.0),
            "number_of_spots": trial.suggest_float("number_of_spots", 0.1, 1.0),
            "distance_from_center": trial.suggest_float("distance_from_center", 0.5, 1.5),
            "visual_novelty": trial.suggest_float("visual_novelty", 0.1, 1.0),
            "heal_reward": trial.suggest_float("heal_reward", 0.5, 1.0),
            "money_reward": trial.suggest_float("money_reward", 0.1, 1.0),
            "died_reward": trial.suggest_float("died_reward", -15.0, -5.0),
            "fainted_reward": trial.suggest_float("fainted_reward", -5.0, -1.0),
            "story_weight": trial.suggest_float("story_weight", 0.5, 1),
            "experience_weight": trial.suggest_float("experience_weight", 0.1, 0.5),
            "exploration_weight": trial.suggest_float("exploration_weight", 0.005, 0.025),
            "tactics_weight": trial.suggest_float("tactics_weight", 0.1, 0.5),
            "text_weight": trial.suggest_float("text_weight", 0.05, 1.0),
        }
        tuner = Tuner(env_config_tuning, ep_length, batch_size, n_epochs, hyperparameter_config, value_function)
        score = tuner.execute_run()
        return score

    def optimize(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        best_params = study.best_params
        best_value = study.best_value
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_value}")
        with open("best_params.json", "w") as f:
            json.dump(best_params, f)


if __name__ == "__main__":
    optimizer = ParameterOptimizer()
    optimizer.optimize()
