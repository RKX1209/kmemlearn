#!/usr/bin/env python

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

FIGDIR="./figures"
if __name__ == '__main__':
    try:
        os.mkdir(FIGDIR)
    except FileExistsError:
        pass

    try:
        with open("data_filtered.pickle", "rb") as pickle_file:
            datasets = pickle.load(pickle_file)
            XL = pickle.load(pickle_file)
    except IOError:
        raise SystemExit("please run prep_data.py before run this script.")
    plt.rcParams["figure.figsize"] = (10,8)
    for di, (filename, label) in enumerate(datasets):
        X = XL[di]
        print(filename, X.shape)
        plt.matshow(np.log(X+1))
        #plt.colorbar()
        plt.title("{} log(histogram)".format(filename))
        plt.xlabel("address index (filtered)")
        plt.ylabel("time series")
        plt.savefig("{}/memtime_{}.png".format(FIGDIR, filename))
    
