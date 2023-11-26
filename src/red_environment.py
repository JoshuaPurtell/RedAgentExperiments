

import uuid 
import os
from math import floor, sqrt
import json
from pathlib import Path

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize


import mediapy as media
import pandas as pd

from gymnasium import Env, spaces
import sys
from pyboy.utils import WindowEvent

from typing import Any, List

from src.device import DeviceHandler
from src.game import GameHandler
from src.reward import RewardHandler
from src.red_types import Reward, VideoHandler, VisualHistoryKNN

from src.globals import col_steps, output_shape, memory_height, mem_padding, output_full, valid_actions, extra_buttons, vec_dim, minimal_reward, agent_save_stats_fields

class RedGymEnv(Env):
    def __init__(
        self, config=None):

        self.config = config

        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.headless = config['headless']
        self.num_elements = 20000 
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.vec_dim = vec_dim
        self.video_interval = 256 * self.act_freq
        self.explore_weight = 1 if 'explore_weight' not in config else config['explore_weight']
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']
        self.similar_frame_dist = config['sim_frame_dist']
        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        use_extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs = []
        self.agent_stats = []
        self.max_steps = config['max_steps']
        self.step_count = 0

        self.output_shape = output_shape
        self.mem_padding = mem_padding
        self.memory_height = memory_height
        self.col_steps = col_steps
        self.output_full = output_full

        if use_extra_buttons:
            valid_actions.extend(extra_buttons)
        self.valid_actions = valid_actions
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)
        self.metadata = {"render.modes": []}
        self.reset()

    def reset(self, seed = None):
        self.seed = seed
        self.seen_coords = {}

        if self.save_video:
            self.video = VideoHandler(self.s_path, self.reset_count, self.instance_id)
            self.devicehandler = DeviceHandler(self.config, valid_actions=self.valid_actions, act_freq=self.act_freq, save_video=self.save_video, fast_video=self.fast_video, headless=self.headless, render_func= self.render, videohandler=self.video)
        else:
            self.devicehandler = DeviceHandler(self.config, valid_actions=self.valid_actions, act_freq=self.act_freq, save_video=self.save_video, fast_video=self.fast_video, headless=self.headless, render_func= self.render)
            
        with open(self.init_state, "rb") as f:
            self.devicehandler.pyboy.load_state(f)
        self.visualhistoryhandler = VisualHistoryKNN(self.vec_dim, self.num_elements, self.similar_frame_dist)
        self.gamehandler = GameHandler(self.devicehandler)
        self.rewardhandler = RewardHandler(reward_scale=self.reward_scale, explore_weight=self.explore_weight)
        return self.render(), {}
    
    def create_exploration_memory_tensor(self, reward: Reward):
        def make_reward_channel(r_val, h=memory_height, w=output_shape[1]):
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered) 
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory
        full_memory = np.stack((
            make_reward_channel(reward.story_reward+reward.experience_reward),
            make_reward_channel(reward.exploration_reward),
            make_reward_channel(reward.tactics_reward)
        ), axis=-1)
        return full_memory
    
    def create_recent_memory_tensor(self,recent_memory):
        recent_memory_tensor = rearrange(
            recent_memory, 
            '(w h) c -> h w c', 
            h=memory_height)
        return recent_memory_tensor
    
    def render(self,reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.devicehandler.get_pixels()
        if reduce_res:
            game_pixels_render = (255*resize(game_pixels_render, output_shape)).astype(np.uint8)
        if update_mem:
            self.gamehandler.history.recent_frames[0] = game_pixels_render
        if add_memory:
            default = Reward()
            default.null()
            last_reward = self.gamehandler.history.rewards[-1] if len(self.gamehandler.history.rewards) > 0 else default
            exploration_memory = self.create_exploration_memory_tensor(last_reward)

            default_last_memory = np.zeros((output_shape[1]*memory_height, 3), dtype=np.uint8)
            last_memory = self.gamehandler.history.recent_memory if len(self.gamehandler.history.recent_memory) > 0 else default_last_memory
            recent_memory_tensor = self.create_recent_memory_tensor(last_memory)
            pad = np.zeros(
                    shape=(mem_padding, output_shape[1], 3), 
                    dtype=np.uint8)
            rf = rearrange(self.gamehandler.history.recent_frames, 'f h w c -> (f h) w c')
            state = np.concatenate(
                    (
                        exploration_memory, 
                        pad,
                        recent_memory_tensor,
                        pad,
                        rf
                    ),
                    axis=0)
        else:
            state = game_pixels_render
        return state

    def roll_over_stm(self,new_prog):
        self.gamehandler.history.recent_memory = np.roll(self.gamehandler.history.recent_memory, 3)
        self.gamehandler.history.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.gamehandler.history.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.gamehandler.history.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

    def check_if_done(self):
        done = self.step_count >= self.max_steps
        return done
    

    def save_and_print_info(self, done: bool, state):
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
            for key, val in [(key, getattr(self.gamehandler.history.rewards[-1], key)) for key in ['story_reward', 'experience_reward', 'exploration_reward', 'tactics_reward']]:
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.gamehandler.history.rewards[-1].total_reward:5.2f}'
        
        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f'curframe_{self.instance_id}.jpeg'), 
                self.render(reduce_res=False, add_memory=False, update_mem=False))

        if self.print_rewards and done:
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.gamehandler.history.rewards[-1].total_reward:.4f}_{self.reset_count}_small.jpeg'), 
                    state)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.gamehandler.history.rewards[-1].total_reward:.4f}_{self.reset_count}_full.jpeg'), 
                    self.render(reduce_res=False, add_memory=False, update_mem=False))

        if self.save_video and done:
            self.devicehandler.videohandler.full_frame_writer.close()
            self.devicehandler.videohandler.model_frame_writer.close()

        if done:
            self.all_runs.append(("\n").join([f'{key}: {val}' for key, val in self.gamehandler.history.rewards[-1].__dict__.items() if key in ['story_reward', 'experience_reward', 'exploration_reward', 'tactics_reward']]))
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')

    def step(self, action):
        self.devicehandler.run_action_on_emulator(action)
        self.gamehandler.update_agent_states(action)
        self.gamehandler.update_seen_coords(self.step_count)
        self.gamehandler.update_raw_texts(self.step_count)
        self.gamehandler.update_texts(self.step_count)
        self.gamehandler.history.recent_frames = np.roll(self.gamehandler.history.recent_frames, -1, axis=0)
        state = self.render()

        frame_start = 2 * (memory_height + mem_padding)
        flat_state = state[frame_start:frame_start+output_shape[0], ...].flatten().astype(np.float32)

        curr_reward, self.visualhistoryhandler = self.rewardhandler.compute_reward(self.gamehandler.history, self.gamehandler,  self.visualhistoryhandler, flat_state)
        self.gamehandler.update_rewards(curr_reward)

        if len(self.gamehandler.history.rewards) > 1:
            total_delta = self.gamehandler.history.rewards[-1].total_reward - self.gamehandler.history.rewards[-2].total_reward
            channel_delta = [self.gamehandler.history.rewards[-1].channel_vec[i] - self.gamehandler.history.rewards[-2].channel_vec[i] for i in range(len(self.gamehandler.history.rewards[-1].channel_vec))]
        else:
            total_delta = minimal_reward 
            channel_delta = np.zeros(3) + minimal_reward 

        self.roll_over_stm(channel_delta)

        done = self.check_if_done()

        self.step_count += 1

        agent_states = self.gamehandler.history.agent_states
        final_info = {attribute: getattr(agent_states[-1], attribute) for attribute in agent_save_stats_fields}
        self.agent_stats.append(final_info)
        self.save_and_print_info(done, state)

        return state, total_delta, False, done, {}