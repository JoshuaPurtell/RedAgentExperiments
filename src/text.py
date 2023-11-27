import hashlib
import json
import os
from pickletools import pybool
#import enchant
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from pyboy import PyBoy
from functools import lru_cache

class TextHandler:
    def __init__(self, pyboy: PyBoy, tiles_cache: Dict[str,Any],hash_to_english: Dict[str,Any]):
        self.pyboy = pyboy
        self.tiles_cache = tiles_cache
        self.hash_to_english = hash_to_english
    
    def get_screen(self):
        screen = self.pyboy.botsupport_manager().screen()
        screen_pixels = screen.screen_ndarray()
        return screen_pixels
    
    def deterministic_hash(self,input_bytes):
        return hashlib.sha256(input_bytes).hexdigest()

    
    @lru_cache(maxsize=None)
    def cached_deterministic_hash(self,tile_bytes):
        return self.deterministic_hash(tile_bytes)
    
    @lru_cache(maxsize=None)
    def cached_get_tile_id(self,tile_bytes):
        return self._get_tile_id(tile_bytes)
    
    def _get_tile_id(self,tilebytes):
        tile_id_counter = len(self.tiles_cache)
        tile_hash = str(self.deterministic_hash(tilebytes))#str(deterministic_hash(tile.tobytes()))
        if tile_hash not in self.tiles_cache:
            self.tiles_cache[tile_hash] = tile_id_counter
        return self.tiles_cache[tile_hash], self.tiles_cache
    
    def get_tiles(self,screen_pixels):
        tile_size = 8
        tiles_id_matrix = []
        tile_hash_matrix = []

        for i in range(0, screen_pixels.shape[0], tile_size):
            row_ids = []
            row_hashes = []
            for j in range(0, screen_pixels.shape[1], tile_size):
                tile = screen_pixels[i : i + tile_size, j : j + tile_size]
                tile_bytes = tile.tobytes()

                tile_hash = self.cached_deterministic_hash(tile_bytes)
                tile_id, tiles_cache  = self.cached_get_tile_id(tile_bytes)

                row_ids.append(tile_id)
                row_hashes.append(tile_hash)
            tiles_id_matrix.append(row_ids)
            tile_hash_matrix.append(row_hashes)
        return tiles_id_matrix, tile_hash_matrix
    
    def get_translation(self,tiles_id_matrix, tiles_hash_matrix):
        translation = []
        tile_ids = []
        for i in range(len(tiles_id_matrix)):
            hash_row = tiles_hash_matrix[i]
            id_row = tiles_id_matrix[i]
            translation_row = []
            for j in range(len(hash_row)):
                if str(hash_row[j]) in self.hash_to_english:
                    translation_row.append(self.hash_to_english[str(hash_row[j])])
                else:
                    translation_row.append("X")
            translation.append(translation_row)
            tile_ids.append(id_row)
        return translation, tile_ids
    
def get_text(pyboy: PyBoy, tiles_cache,hash_to_english):
    text_handler = TextHandler(pyboy, tiles_cache,hash_to_english)
    screen_pixels = text_handler.get_screen()
    tiles_id_matrix, tiles_hash_matrix = text_handler.get_tiles(screen_pixels)
    translation, tile_ids = text_handler.get_translation(tiles_id_matrix, tiles_hash_matrix)
    sequential_translation = "\n".join(
        [
            "".join(row) 
            for row in translation 
            if len(set(["X", " "]).intersection(set(row))) < len(row)
        ]
    )
    return sequential_translation, text_handler.tiles_cache
