import os
import csv
import sys

import click
import hickle
import keras
import numpy as np
from deepyeast.dataset import load_transfer_data
from deepyeast.models import DeepYeast
from deepyeast.utils import preprocess_input
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 64

NUM_CLASSES = 28


@click.command(help="Create dataset.")
@click.option("-i", "--images-path", prompt=True, type=str)
@click.option("-l", "--labels-path", prompt=True, type=str)
def main(
    images_path,
    labels_path,
):  
    paths  = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(".png")]
    create_dataset_rg(paths, labels_path)


# load_dataset_rg loads a RG dataset.
# imgs_path: the paths of the images
# label_csv_path: the csv with the dataset labels, two columns: id, labels are expected.
def create_dataset_rg(imgs_paths, label_csv_path, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
    with open("training_idxs.hkl") as f:
        training_idxs = hickle.load(f)

    with open("val_idxs.hkl") as f:
        validation_idxs = hickle.load(f)

    training_idxs.remove("ad17b4f6-bba8-11e8-b2ba-ac1f6b6435d0")

    
    train = np.memmap("training.dat", dtype="float32", mode='w+', shape=(510737, width, height, 2))
    train_labels = np.memmap("training_labels.dat", dtype="float32", mode='w+', shape=(510737, 28))

    val = np.memmap("validation.dat", dtype="float32", mode='w+', shape=(127979, width, height, 2))
    val_labels = np.memmap("validation_labels.dat", dtype="float32", mode='w+', shape=(127979, 28))

    input_label_file = csv.DictReader(open(label_csv_path))
    ids_to_labels = _get_images_ids_to_labels(input_label_file)
    ids_to_paths = _get_images_ids_to_paths(imgs_paths)

    train_idx_list = []
    val_idx_list = []

    i = 0
    pbar = tqdm(desc="Loading images")
    for img_id, path in ids_to_paths.items():
        if img_id in training_idxs:
            train_idx_list.append(img_id)
            if isinstance(path, list):
                for p in path:
                    img = load_img(p)
                    img_array = img_to_array(img)
                    train[i, :, :]  = img_array[:, :, :2]
                    train_labels[i, :] = ids_to_labels[img_id]
                    i += 1
                    pbar.update(1)
            else:
                img = load_img(path)
                img_array = img_to_array(img)
                train[i, :, :]  = img_array[:, :, :2]
                train_labels[i, :] = ids_to_labels[img_id]
                i += 1
                pbar.update(1)

    pbar.close()

    i = 0
    pbar = tqdm(desc="Loading images")
    for img_id, path in ids_to_paths.items():
        if img_id in validation_idxs:
            val_idx_list.append(img_id)
            if isinstance(path, list):
                for p in path:
                    img = load_img(p)
                    img_array = img_to_array(img)
                    val[i, :, :]  = img_array[:, :, :2]
                    val_labels[i, :] = ids_to_labels[img_id]
                    i += 1
                    pbar.update(1)
            else:
                img = load_img(path)
                img_array = img_to_array(img)
                val[i, :, :]  = img_array[:, :, :2]
                val_labels[i, :] = ids_to_labels[img_id]
                i += 1
                pbar.update(1)

    pbar.close()

    with open("train_actual_idxs.hkl", "w") as f:
        hickle.dump(train_idx_list, f)
    
    with open("val_actual_idxs.hkl", "w") as f:
        hickle.dump(val_idx_list, f)


def _get_dim(idxs, file_list):
    dim = 0
    for f in file_list:
        for idx in idxs:
            if idx in f:
                dim += 1
                break

    return dim


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

        if "_" in base:
            img_id = base.split("_")[0]
            if img_id in ids_to_paths:
                ids_to_paths[img_id].append(path)
            else:
                ids_to_paths[img_id] = [path]
        else:
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


if __name__ == "__main__":
    main()
