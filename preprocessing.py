import os
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

import numpy as np
from PIL import Image

RGB_CHANNEL_TO_INDEX = {"red": 0, "green": 1, "blue": 2}
NUM_PROCESSES = 4


def get_images_prefixes(images_paths: list) ->list:
    """
    >>> get_images_prefixes(["one-path_aaa.png", "another-path_aaa.png", "one-path_bbb.png"])
    >>> ['one-path', 'another-path']
    >>> get_images_prefixes(["this-path_is-not_fine"])
    Traceback (most recent call last):
    ...
    ValueError: uknown path format 'this-path_is-not_fine'
    """
    prefixes = set()
    for path in images_paths:
        split_path = path.split("_")
        if len(split_path) != 2:
            raise ValueError("uknown path format %s", path)
        prefixes.add(split_path[0])

    return list(prefixes)
        
    
def get_rgb_to_prefixes(prefixes: list, width: int, height: int) -> dict:
    prefixes_to_rgb = {}

    def f(prefix):
        prefixes_to_rgb[prefix] = get_rgb(prefix, width, height)
    
    pool = ThreadPool(processes=NUM_PROCESSES)
    with tqdm(total=len(prefixes), desc="creating rgb images.", unit="images") as pbar:
        for i, _ in tqdm(enumerate(pool.imap(f, prefixes))):
            pbar.update()

    return prefixes_to_rgb


def get_rg_to_prefixes(prefixes: list, width: int, height: int) -> dict:
    prefixes_to_rg = {}

    def f(prefix):
        prefixes_to_rg[prefix] = get_rg(prefix, width, height)
    
    pool = ThreadPool(processes=NUM_PROCESSES)
    with tqdm(total=len(prefixes), desc="creating rg images.", unit="images") as pbar:
        for i, _ in tqdm(enumerate(pool.imap(f, prefixes))):
            pbar.update()

    return prefixes_to_rg


def get_rgb(prefix: str, width, height) -> np.ndarray:
    rgb_image = np.zeros(shape=(height, width, 3), dtype=np.float)

    for channel, idx in RGB_CHANNEL_TO_INDEX.items():
        current_image = Image.open(prefix + "_" + channel + ".png")
        rgb_image[:, :, idx] = current_image
    
    rgb_image = rgb_image / rgb_image.max() * 255
    return rgb_image.astype(np.uint8)


def get_rg(prefix: str, width, height) -> np.ndarray:
    rg_image = np.zeros(shape=(height, width, 3), dtype=np.float)

    for channel, idx in RGB_CHANNEL_TO_INDEX.items():
        current_image = Image.open(prefix + "_" + channel + ".png")
        if channel != "green":
            # All the non-green channels are mapped to the red channel.
            rg_image[:, :, RGB_CHANNEL_TO_INDEX["red"]] += current_image
        else:
            rg_image[:, :, idx] += current_image
    
    rg_image = rg_image / rg_image.max() * 255
    return rg_image.astype(np.uint8)

