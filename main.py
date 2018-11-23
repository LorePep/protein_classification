import argparse
import os
import sys
from tqdm import tqdm
from multiprocessing import Pool

from PIL import Image

from preprocessing import (
    get_images_prefixes,
    get_rgb_to_prefixes)


DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512


def main():
    parser = argparse.ArgumentParser(description="Preprocess images.")
    parser.add_argument("input", type=str, help="dataset input path")
    parser.add_argument("--output-path", "-o", type=str, help="preprocessing output path", required=True)
    parser.add_argument("--create-rgb", dest="create", action="store_true",
                        help="create rgb for images")

    args = parser.parse_args()
    validate_args(args)

    if args.create:
        create_rgbs(args.input)
    else:
        print("Nothing to do.")


def create_rgbs(input_path: str):
    paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".png")]
    prefixes = get_images_prefixes(paths)
    prefixes_to_rgb = get_rgb_to_prefixes(prefixes, DEFAULT_WIDTH, DEFAULT_HEIGHT)
    
    for prefix, rgb in tqdm(prefixes_to_rgb.items(), desc="saving images on disk", unit="images"):
        im = Image.fromarray(rgb)
        base = os.path.basename(prefix)
        rgb_path = os.path.join(args.output_path, base + ".png")
        im.save(rgb_path)


def validate_args(args):
    if not os.path.isdir(args.output_path):
        print("Specified output path {} is not a directory".format(args.output_path))
        sys.exit()
    

if __name__ == "__main__":
    main()


