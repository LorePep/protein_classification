import os
import csv
import sys

import keras
import numpy as np
from deepyeast.dataset import load_transfer_data
from deepyeast.models import DeepYeast
from deepyeast.utils import preprocess_input
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

NUM_CLASSES = 28

# load_dataset_rg loads a RG dataset.
# imgs_path: the paths of the images
# label_csv_path: the csv with the dataset labels, two columns: id, labels are expected.
def load_dataset_rg(imgs_paths, label_csv_path, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
    data = np.empty((len(imgs_paths), width, height, 2))
    labels = np.empty((len(imgs_paths), 28))

    input_label_file = csv.DictReader(open(label_csv_path))
    ids_to_labels = _get_images_ids_to_labels(input_label_file)
    ids_to_paths = _get_images_ids_to_paths(imgs_paths)

    i = 0
    for img_id, path in tqdm(ids_to_paths.items(), desc="Loading images", total=len(ids_to_paths)):
        img = load_img(path)
        img_array = img_to_array(img)
        data[i, :, :] = img_array[:, :, :2]
        labels[i, :] = ids_to_labels[img_id]
        i += 1

    return data, labels


def _get_images_ids_to_labels(input_file):
    ids_to_labels = {}

    for row in input_file:
        img_id = row["Id"]
        label = _parse_label(row["Target"])
        ids_to_labels[img_id] = label

    return ids_to_labels


def _get_images_ids_to_paths(imgs_paths):
    ids_to_paths = {}

    for path in imgs_paths:
        base = os.path.basename(path)
        if not base.endswith(".png"):
            raise ValueError("file format not expected")

        img_id = base[:-4]
        ids_to_paths[img_id] = path

    return ids_to_paths


def _parse_label(target):
    split_target = target.split(" ")
    
    if len(split_target) < 1:
        raise ValueError("unknown target format")
    
    label = np.zeros((1, NUM_CLASSES))

    for l in split_target:
        label[0, int(l)] = 1    
    
    return label
