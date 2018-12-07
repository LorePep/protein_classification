import logging
import os

import click
import keras
import hickle as hkl 
import numpy as np
from deepyeast.models import DeepYeast
from sklearn.metrics import fbeta_score
from sklearn import model_selection
from scipy.optimize import fmin_l_bfgs_b, basinhopping
from keras import Model
from keras.layers import Dense
from sklearn.metrics import f1_score
from tabulate import tabulate
from timeit import default_timer as timer


logging.getLogger().setLevel(logging.INFO)

@click.command(help="Analyse model results.")
@click.option("-w", "--weights-path", prompt=True, type=str)
def main(
    weights_path: str,
) -> None:
    logging.info("Loading base model")
    base_model = DeepYeast()
    logging.info("Changing model")
    relu5_features = base_model.get_layer("relu5").output
    
    scores = Dense(28, activation="sigmoid")(relu5_features)
    model = Model(inputs=base_model.input, outputs=scores)

    model.load_weights(weights_path)

    logging.info("loading data")
    X = np.memmap("dataset.dat", dtype='float32', mode='r', shape=(638716, 64, 64, 2))
    y = np.memmap("labels.dat", dtype='float32', mode='r', shape=(638716, 28))

    X_train, X_val_test, y_train, y_val_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    X_val, X_test, y_val, y_test = model_selection.train_test_split(X_val_test, y_val_test, test_size=0.5, shuffle=True, random_state=42)
    
    X_train /= 255.
    X_train -= 0.5
    X_train *= 2.

    X_val /= 255.
    X_val -= 0.5
    X_val *= 2. 
    

    # logging.info("running training set")
    # y_pred_train = model.predict(X_train, batch_size=1024, verbose=1, steps=None)
    # logging.info("running validation set")
    y_pred_val = model.predict(X_val, batch_size=1014, verbose=1, steps=None)
    print(y_pred_val.shape)

    # f1_train, _ = best_f2_score(y_train, y_pred_train)
    f1_val, x_val = best_f2_score(y_val, y_pred_val)
    print(f1_val, x_val)    

    # print(tabulate([["Training", "{0:.2f}".format(f1_train)],
    #      ["Validation", "{0:.2f}".format(f1_val)]], headers=["Dataset", "F1_macro"]))



def best_f2_score(true_labels, predictions):

    def f_neg(threshold):
        ## Scipy tries to minimize the function so we must get its inverse
        return - fbeta_score(true_labels, predictions > threshold, beta=2, average='samples')

    # Initialization of best threshold search
    thr_0 = [0.20] * true_labels.shape[1]
    constraints = [(0.,1.)] * true_labels.shape[1]
    def bounds(**kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= 1))
        tmin = bool(np.all(x >= 0)) 
        return tmax and tmin
    
    # Search using L-BFGS-B, the epsilon step must be big otherwise there is no gradient
    minimizer_kwargs = {"method": "L-BFGS-B",
                        "bounds":constraints,
                        "options":{
                            "eps": 0.05
                            }
                       }
    
    # We combine L-BFGS-B with Basinhopping for stochastic search with random steps
    print("===> Searching optimal threshold for each label")
    start_time = timer()
    
    opt_output = basinhopping(f_neg, thr_0,
                                stepsize = 0.1,
                                minimizer_kwargs=minimizer_kwargs,
                                niter=10,
                                accept_test=bounds)
    
    end_time = timer()
    print("===> Optimal threshold for each label:\n{}".format(opt_output.x))
    print("Threshold found in: %s seconds" % (end_time - start_time))
    
    score = - opt_output.fun
    return score, opt_output.x


if __name__ == "__main__":
    main()    
