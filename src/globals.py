from pyboy.utils import WindowEvent

ep_length = 2048#2048 * 10
batch_size = 64
n_epochs = 3

col_steps = 16
output_shape = (36, 40, 3)
memory_height = 8
mem_padding = 2
frame_stacks = 3
reward_range = (0, 15000)
output_full = (output_shape[0] * frame_stacks + 2 * (mem_padding + memory_height), output_shape[1], output_shape[2])
vec_dim = 4320

#print_rewards = False
#"sim_frame_dist": 2_000_000.0
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
    "sim_frame_dist": 20_000.0,
    "use_screen_explore": True,
    "reward_scale": 4,
    "extra_buttons": False,
    "explore_weight": 3,  # 2.5
}

valid_actions = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
]
        
extra_buttons = [
    WindowEvent.PRESS_BUTTON_START,
    WindowEvent.PASS
]