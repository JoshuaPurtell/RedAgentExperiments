from tabnanny import check
from typing import Any, Dict
import numpy as np
import optuna
import json
import os

from experiments.hyperparam_tuning import Tuner, BaseRun


def value_function(env_infos: Dict[str, Any]) -> float:
    badges = np.mean([env_info["badges"] for env_info in env_infos])
    avg_levels = np.mean([np.mean([level for level in env_info["levels"] if level > 0]) for env_info in env_infos])
    seen_pokemon = np.mean([env_info["n_pokemon_seen"] for env_info in env_infos])
    op_level = np.mean([env_info["op_level"] for env_info in env_infos])
    u_texts_scene = np.mean([len(set([t[0] for t in env_info["texts_seen"]])) for env_info in env_infos])
    return 10*badges + 0.01 * avg_levels + 0.05 * seen_pokemon + 0.03 * op_level + .00001 * u_texts_scene


class BaselineBuilder:
    def __init__(self, checkpoint_file = None):
        self.base_ep_length =  128*20# 204, 2048* 10
        self.base_batch_size = 32
        self.base_n_epochs = 3
        self.hyperparameter_config = {
            "badge_reward": 1000,
            "seen_pokemon_reward": 50.0,
            "op_level_reward": .5,
            "levels_reward": 10,
            "p_types_reward": .5,
            "rel_number_of_times_weve_been_here": -3,
            "number_of_spots": 2,
            "distance_from_center": 2,
            "visual_novelty": 3,
            "heal_reward": .8,
            "money_reward": .5,
            "died_reward": -10,
            "fainted_reward": -2,
            "story_weight": .75,
            "experience_weight": .25,
            "exploration_weight": .025,
            "tactics_weight": .2,
            "text_weight": .1,
        }
        self.base_env_config = {
            "headless": True,
            "save_final_state": True,
            "early_stop": False,
            "action_freq": 24,
            "init_state": "storage/has_pokedex_nballs_copy.state",
            "max_steps": self.base_ep_length,
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
        self.checkpoint_file = checkpoint_file
    
    def optimize(self):
        baserunner = BaseRun(self.base_env_config, self.base_ep_length, self.base_batch_size, self.base_n_epochs, self.hyperparameter_config, value_function)
        checkpoint_file = baserunner.execute_run()
        return checkpoint_file


class ParameterOptimizer:
    def __init__(self, n_trials: int = 1, checkpoint_file: str = None):
        self.tune_ep_length = 128*12
        self.tune_batch_size = 32
        self.tune_n_epochs = 2

        self.n_trials = n_trials
        self.checkpoint_file = checkpoint_file

    def objective(self, trial: optuna.Trial) -> float:
        self.hyperparameter_config = {
            "badge_reward": trial.suggest_float("badge_reward", 500.0, 1500.0),
            "seen_pokemon_reward": trial.suggest_float("seen_pokemon_reward", 1.0, 3.0),
            "op_level_reward": trial.suggest_float("op_level_reward", 0.1, 1.5),
            "levels_reward": trial.suggest_float("levels_reward", 0.1, 1.5),
            "p_types_reward": trial.suggest_float("p_types_reward", 0.1, 1.5),
            "rel_number_of_times_weve_been_here": trial.suggest_float("rel_number_of_times_weve_been_here", -1.0, 0.0),
            "number_of_spots": trial.suggest_float("number_of_spots", 0.01, .3),
            "distance_from_center": trial.suggest_float("distance_from_center", 0.1, 0.5),
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
        self.tune_env_config = {
            "headless": True,
            "save_final_state": True,
            "early_stop": False,
            "action_freq": 24,
            "init_state": "storage/has_pokedex_nballs_copy.state",
            "max_steps": self.tune_ep_length,
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
        tuner = Tuner(self.tune_env_config, self.tune_ep_length, self.tune_batch_size, self.tune_n_epochs, self.hyperparameter_config, value_function,checkpoint_file=self.checkpoint_file)
        score = tuner.execute_run()
        return score

    def optimize(self):
        if os.path.exists("best_params.json"):
            with open("best_params.json") as f:
                current_best_params = json.load(f)
        else:
            current_best_params = {}
            current_best_params["value"] = -np.inf
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        run_best_params = study.best_params
        best_value = study.best_value
        run_best_params["value"] = best_value
        print(f"Best parameters: {run_best_params}")
        print(f"Best score: {best_value}")
        print(f"""Previous best score: {current_best_params["value"]}""")
        if best_value > current_best_params["value"]:
            with open("best_params.json", "w") as f:
                json.dump(run_best_params, f)
        print("Done")

if __name__ == "__main__":
    compound_base = False
    checkpoint_file = None
    if os.path.exists("storage/sessions/base"):
        runs = os.listdir("storage/sessions/base")
        if len(runs)> 0:
            checkpoint_file = [r for r in reversed(runs)][0]
    print(f"Checkpoint file: {checkpoint_file}")
    if not checkpoint_file:
        print("Building baseline")
        baseline = BaselineBuilder()
        checkpoint_file = baseline.optimize()
    elif compound_base:
        print("Compounding baseline")
        baseline = BaselineBuilder(checkpoint_file=checkpoint_file)
        checkpoint_file = baseline.optimize()
    optimizer = ParameterOptimizer(checkpoint_file=checkpoint_file,n_trials = 15)
    optimizer.optimize()
    print("Here")
