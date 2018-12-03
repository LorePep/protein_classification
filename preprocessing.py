import argparse
import os
import sys
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
from PIL import Image
from tqdm import tqdm

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
RGB_CHANNEL_TO_INDEX = {"red": 0, "green": 1, "blue": 2}
NUM_PROCESSES = 8
NUMBER_OF_SUBIMAGES = 30


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess images.")
    parser.add_argument("input", type=str, help="dataset input path")
    parser.add_argument("--output-path", "-o", type=str, help="preprocessing output path", required=True)
    parser.add_argument("--create-rgb", dest="create_rgbs", action="store_true",
                        help="create rgb for images")
    parser.add_argument("--create-rg", dest="create_rgs", action="store_true",
                        help="create rg for images")
    parser.add_argument("--create-windows", dest="create_win", action="store_true",
                        help="create windowed dataset for images")
    parser.add_argument("--discard-images", dest="discard_bad", action="store_true",
                        help="filter bad images")

    args = parser.parse_args()
    validate_args(args)

    if args.create_rgbs:
        create_rgbs(args.input, args.output_path)
    elif args.create_rgs:
        create_rgs(args.input, args.output_path)
    elif args.create_win:
        create_win(args.input, args.output_path)
    elif args.discard_bad:
        discard_bad_images(args.input, args.output_path)
    else:
        print("Nothing to do.")


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


def create_rgbs(input_path: str, output_path: str) -> None:
    paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".png")]
    prefixes = get_images_prefixes(paths)
    prefixes_to_rgb = get_rgb_to_prefixes(prefixes, DEFAULT_WIDTH, DEFAULT_HEIGHT)

    for prefix, rgb in tqdm(prefixes_to_rgb.items(), desc="saving images on disk", unit="images"):
        im = Image.fromarray(rgb)
        base = os.path.basename(prefix)
        rgb_path = os.path.join(output_path, f"{base}.png")
        im.save(rgb_path)


def create_rgs(input_path: str, output_path: str, img_size: int = 512) -> None:
    paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".png")]
    prefixes = get_images_prefixes(paths)
    prefixes_to_rg = get_rg_to_prefixes(prefixes, DEFAULT_WIDTH, DEFAULT_HEIGHT)
    
    for prefix, rgb in tqdm(prefixes_to_rg.items(), desc="saving images on disk", unit="images"):
        im = Image.fromarray(rgb).resize((img_size, img_size), Image.ANTIALIAS)
        base = os.path.basename(prefix)
        rg_path = os.path.join(output_path, f"{base}.png")
        im.save(rg_path)

def create_win(input_path: str, output_path: str, win_size: int = 64) -> None:
    paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".png")]
    pool = ThreadPool(processes=NUM_PROCESSES)

    def f(path):
        base = os.path.basename(path)
        img = Image.open(path)
    
        for i in range(NUMBER_OF_SUBIMAGES):
            im =  Image.fromarray(select_random_window(np.array(img), win_size))
            win_path = os.path.join(output_path, "win",f"{base[:-4]}_{i}.png")
            im.save(win_path)
        
    
    with tqdm(total=len(paths), desc="creating windowed images.", unit="images") as pbar:
        for i, _ in tqdm(enumerate(pool.imap(f, paths))):
            pbar.update()



def discard_bad_images(input_path: str, output_path: str) -> None:
    paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".png")]
    pool = ThreadPool(processes=NUM_PROCESSES)

    def f(path):
        base = os.path.basename(path)
        img = Image.open(path)

        if is_image_good(np.array(img)): 
            new_img_path = os.path.join(output_path, "filt", base)
            img.save(new_img_path)
        
    
    with tqdm(total=len(paths), desc="creating windowed images.", unit="images") as pbar:
        for i, _ in tqdm(enumerate(pool.imap(f, paths))):
            pbar.update()


def is_image_good(img: np.ndarray) -> bool:
    return img.mean() > 5


def validate_args(args):
    if not os.path.isdir(args.output_path):
        print("Specified output path {} is not a directory".format(args.output_path))
        sys.exit()
    

def select_random_window(arr, window_size):
    offset = np.random.randint(0, arr.shape[1]-window_size+1)
    return arr[offset:offset+window_size, offset:offset+window_size]


if __name__ == "__main__":
    main()
