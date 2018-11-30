import logging
import os

import click
import keras
import hickle as hkl 
import numpy as np
from deepyeast.models import DeepYeast
from keras import Model
from keras.layers import Dense
from sklearn.metrics import f1_score
from tabulate import tabulate


logging.getLogger().setLevel(logging.INFO)

@click.command(help="Analyse model results.")
@click.option("-w", "--weights-path", prompt=True, type=str)
@click.option("-t", "--training-path", prompt=True, type=str)
@click.option("-v", "--validation-path", prompt=True, type=str)
def main(
    weights_path: str,
    training_path: str,
    validation_path: str) -> None:
    logging.info("Loading base model")
    base_model = DeepYeast()
    logging.info("Changing model")
    relu5_features = base_model.get_layer("relu5").output
    
    scores = Dense(28, activation="sigmoid")(relu5_features)
    model = Model(inputs=base_model.input, outputs=scores)

    model.load_weights(weights_path)

    logging.info("loading data")
    X_train, y_train = hkl.load(training_path)
    X_val, y_val = hkl.load(validation_path)

    logging.info("running training set")
    y_pred_train = model.predict(X_train, batch_size=2000, verbose=1, steps=None)
    logging.info("running validation set")
    y_pred_val = model.predict(X_val, batch_size=500, verbose=1, steps=None)

    y_pred_train[y_pred_train > 0.5] = 1
    y_pred_train[y_pred_train < 0.5] = 0
    y_pred_val[y_pred_val > 0.5] = 1
    y_pred_val[y_pred_val < 0.5] = 0

    f1_train = f1_score(y_train, y_pred_train, average="macro")
    f1_val = f1_score(y_val, y_pred_val, average="macro")

    print(tabulate([["Training", "{0:.2f}".format(f1_train)],
         ["Validation", "{0:.2f}".format(f1_val)]], headers=["Dataset", "F1_macro"]))


if __name__ == "__main__":
    main()    
