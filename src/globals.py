from pyboy.utils import WindowEvent

ep_length = 2048 * 10  # 204, 2048* 10
batch_size = 64
n_epochs = 3

save_freq = 500

col_steps = 16
output_shape = (36, 40, 3)
memory_height = 8
mem_padding = 2
frame_stacks = 3
reward_range = (0, 15000)
output_full = (output_shape[0] * frame_stacks + 2 * (mem_padding + memory_height), output_shape[1], output_shape[2])
vec_dim = 4320
minimal_reward = 0.00000000001
agent_save_stats_fields = [
    "money",
    "badges",
    "levels",
    "hps",
    "max_hps",
    "seen_pokemon",
    "op_level",
    "ptypes",
    "map_location",
    "map_n",
    "x_pos",
    "y_pos",
]

# "sim_frame_dist": 2_000_000.0
env_config = {
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

reward_hyperparameters = {
    "badge_reward": 100.0,
    "seen_pokemon_reward": 1.0,
    "op_level_reward": 1.0,
    "levels_reward": 1.0,
    "p_types_reward": 1.0,
    "rel_number_of_times_weve_been_here": 1.0,
    "number_of_spots": 1.0,
    "distance_from_center": 1.0,
    "visual_novelty": 1.0,
    "heal_reward": 1.0,
    "money_reward": 1.0,
    "died_reward": 1.0,
    "fainted_reward": 1.0,
    "story_weight": 0.25,
    "experience_weight": 0.25,
    "exploration_weight": 0.25,
    "tactics_weight": 0.25,
    "text_weight": 0.05,
}

valid_actions = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
]

extra_buttons = [WindowEvent.PRESS_BUTTON_START, WindowEvent.PASS]
