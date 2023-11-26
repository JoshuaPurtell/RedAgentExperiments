import hashlib
import json
import os
from pickletools import pybool
#import enchant
import time

import numpy as np
from PIL import Image
from pyboy import PyBoy
from functools import lru_cache

def get_text(pyboy: PyBoy):
    screen_pixels = get_screen(pyboy)
    tiles_id_matrix, tiles_hash_matrix = get_tiles(screen_pixels)
    translation, tile_ids = get_translation(tiles_id_matrix, tiles_hash_matrix)
    sequential_translation = "\n".join(
        [
            "".join(row) 
            for row in translation 
            if len(set(["X", " "]).intersection(set(row))) < len(row)
        ]
    )
    return sequential_translation

def deterministic_hash(input_bytes):
    return hashlib.sha256(input_bytes).hexdigest()


def get_screen(pyboy):
    screen = pyboy.botsupport_manager().screen()
    screen_pixels = screen.screen_ndarray()
    return screen_pixels

@lru_cache(maxsize=None)
def cached_deterministic_hash(tile_bytes):
    return deterministic_hash(tile_bytes)

@lru_cache(maxsize=None)
def cached_get_tile_id(tile_bytes):
    return _get_tile_id(tile_bytes)

def get_tiles(screen_pixels):
    tile_size = 8
    tiles_id_matrix = []
    tile_hash_matrix = []

    for i in range(0, screen_pixels.shape[0], tile_size):
        row_ids = []
        row_hashes = []
        for j in range(0, screen_pixels.shape[1], tile_size):
            tile = screen_pixels[i : i + tile_size, j : j + tile_size]
            tile_bytes = tile.tobytes()

            tile_hash = cached_deterministic_hash(tile_bytes)
            tile_id = cached_get_tile_id(tile_bytes)

            row_ids.append(tile_id)
            row_hashes.append(tile_hash)
        tiles_id_matrix.append(row_ids)
        tile_hash_matrix.append(row_hashes)

    return tiles_id_matrix, tile_hash_matrix


def _get_tile_id(tilebytes, tiles_cache_path="run/tiles_cache.json"):
    with open(tiles_cache_path) as f:
        tiles_cache = json.load(f)
    tile_id_counter = len(tiles_cache)
    tile_hash = str(deterministic_hash(tilebytes))#str(deterministic_hash(tile.tobytes()))
    if tile_hash not in tiles_cache:
        tiles_cache[tile_hash] = tile_id_counter
        with open(tiles_cache_path, "w") as f:
            json.dump(tiles_cache, f)
    return tiles_cache[tile_hash]


def get_translation(tiles_id_matrix, tiles_hash_matrix, hash_to_english_path="run/hash_to_english.json"):
    with open(hash_to_english_path) as f:
        hash_to_english = json.load(f)
    translation = []
    tile_ids = []
    for i in range(len(tiles_id_matrix)):
        hash_row = tiles_hash_matrix[i]
        id_row = tiles_id_matrix[i]
        translation_row = []
        for j in range(len(hash_row)):
            if str(hash_row[j]) in hash_to_english:
                translation_row.append(hash_to_english[str(hash_row[j])])
            else:
                translation_row.append("X")
        translation.append(translation_row)
        tile_ids.append(id_row)
    return translation, tile_ids


def dump_text(pyboy: PyBoy, tile_hash_map_path="run/hash_to_english.json"):
    screen_pixels = get_screen(pyboy)
    tiles_id_matrix, tiles_hash_matrix = get_tiles(screen_pixels)
    translation, tile_ids = get_translation(
        tiles_id_matrix, tiles_hash_matrix, hash_to_english_path=tile_hash_map_path
    )

    valid = False
    for i in range(len(translation)):
        non_X = [translation[i][j] for j in range(len(translation[i])) if translation[i][j] not in ["X", " "]]
        X = [translation[i][j] for j in range(len(translation[i])) if translation[i][j] in ["X"]]
        if len(non_X) > 0 and len(X) > 0 and "POKe'DEX" not in translation[i]:
            valid = True
            break
    if valid:
        dump_files = os.listdir("run/text_dumps")
        count_dumps = len([file for file in dump_files if "dump_" in file])
        with open(f"run/text_dumps/dump_{count_dumps}.txt", "a") as f:
            f.write("New dump\n")
            f.write("Raw Translation: ")
            f.write("\n".join([f"{i}: {str(translation[i])}" for i in range(len(translation))]))
            f.write("\nTile ids: ")
            f.write("\n".join([f"{i}: {str(tile_ids[i])}" for i in range(len(tile_ids))]))
            f.write("\n")
            image = Image.fromarray(screen_pixels)
            image.save(f"run/image_dumps/screen_pixels_{count_dumps}.png")