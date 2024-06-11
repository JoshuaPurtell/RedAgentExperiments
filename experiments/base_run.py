from experiments.hyperparam_tuning import BaseRun
if __name__=="__main__":
    tune_ep_length = 1280*12
    tune_batch_size = 32
    tune_n_epochs = 2
    hyperparameter_config = {
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
    tune_env_config = {
            "headless": True,
            "save_final_state": True,
            "early_stop": False,
            "action_freq": 24,
            "init_state": "storage/has_pokedex_nballs_copy.state",
            "max_steps": tune_ep_length,
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
    base = BaseRun(tune_env_config, tune_ep_length, tune_batch_size, tune_n_epochs, hyperparameter_config, None,checkpoint_file="storage/sessions/base/session_2024-02-02-23-44-53/poke_92160_steps")
    path = base.execute_run()
    print(path)