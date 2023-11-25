from pyboy.utils import WindowEvent
from pyboy import PyBoy
from pyboy.logger import log_level
from gymnasium import Env, spaces
import sys
import numpy as np

from src.globals import col_steps, output_shape, memory_height, mem_padding, frame_stacks
from src.red_types import VideoHandler

class DeviceHandler:
    def __init__(self,config=None, valid_actions=[], act_freq=24, save_video=False, fast_video=False, headless=False, render_func = None, videohandler: VideoHandler = None):
        self.config = config
        self.render_func = render_func
        self.valid_actions = valid_actions

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]
        self.output_full = (
            output_shape[0] * frame_stacks + 2 * (mem_padding + memory_height),
                            output_shape[1],
                            output_shape[2]
        )

        head = 'headless' if config['headless'] else 'SDL2'

        log_level("ERROR")
        self.pyboy = PyBoy(
                config['gb_path'],
                debugging=True,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
            )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config['headless']:
            self.pyboy.set_emulation_speed(6)
        self.act_freq = act_freq
        self.save_video = save_video
        self.fast_video = fast_video
        self.headless = headless
        if self.save_video:
            self.videohandler = videohandler

    def add_video_frame(self):
        self.videohandler.full_frame_writer.add_image(self.render_func(reduce_res=False, update_mem=False))
        self.videohandler.model_frame_writer.add_image(self.render_func(reduce_res=True, update_mem=False))
    
    def run_action_on_emulator(self, action: int):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq-1:
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()
    
    def get_pixels(self):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        return game_pixels_render
    
    def read_m(self, addr: int) -> int:
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr: int, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit-1] == '1'
    
    def bit_count(self, bits: int) -> int:
        return bin(bits).count('1')