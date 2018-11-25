import os
import sys

import keras
import numpy as np

from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import (
    img_to_array,
    load_img)

from deepyeast.dataset import load_transfer_data
from deepyeast.models import DeepYeast
from deepyeast.utils import preprocess_input

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512


def load_dataset_rg(input_path: str) -> np.ndarray:
    imgs_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".png")]
    # We assume the imput data to be RG
    data = np.empty((len(imgs_paths), DEFAULT_WIDTH, DEFAULT_HEIGHT, 2))
    print(data.shape)

    for i, path in enumerate(imgs_paths):
        img = load_img(path)
        img_array = img_to_array(img)
        data[i, :, :] = img_array[:, :, :2]

    return data

