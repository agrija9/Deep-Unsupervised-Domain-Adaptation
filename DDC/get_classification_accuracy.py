#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pathlib import Path
import argparse
import statistics

"""
Created on Friday Mar 20 2020
@authors: Alan Preciado, Santosh Muthireddy
"""
def get_classification_accuracy(source, target, no_epochs):

    # specify path where in log folder where training logs are saved
    pkldir = os.path.join("logs", source + "_to_" + target,
                          str(no_epochs) + "_epochs_128_s_128_t_batch_size")

    if Path(pkldir).is_dir():
        print("directory with pkl files is:", pkldir)

    else:
        print("folder with pkl data does not exist, must train model")
        return None

    # load dictionaries with log information
    path_adapt_log = [pkldir + "/adaptation_testing_t_statistic.pkl"]

    print(">>>Loading pkl files<<<")
    adapt_testing_target_dict = pickle.load(open(path_adapt_log[0], 'rb'))
    print(">>>pkl files loaded correctly<<<")

    # create dictionary structures for adaptation and no-adaptation results

    # (w coral loss)
    adaptation = {
        "classification_loss": [],
        "ddc_loss": [],
        "source_accuracy": [],
        "target_accuracy": []
    }

    # get average coral and classification loss for steps in each epoch
    # get accuracy obtained in each epoch
    for epoch_idx in range(len(adapt_testing_target_dict)): # epochs
        # store accuracies in adaptation dictionary
        # adaptation["source_accuracy"].append(adapt_testing_source_dict[epoch_idx]["accuracy %"])
        adaptation["target_accuracy"].append(adapt_testing_target_dict[epoch_idx]["accuracy %"])

    print(">>>Classification accuracies on target domain<<<")
    print(adaptation["target_accuracy"])

    print(">>>Object recognition accuracy (mean) for {0} source and {1} target<<<".format(source, target))
    avg_accuracy = statistics.mean(adaptation["target_accuracy"])
    std_deviation = statistics.stdev(adaptation["target_accuracy"])
    print(avg_accuracy)
    print(std_deviation)


    # plot accuracies for test data in source and target domains
    fig=plt.figure(figsize=(8, 6), dpi=100)
    fig.show()

    plt.xlabel("epochs", fontsize=15)
    plt.ylabel("classification accuracy (%)", fontsize=15)
    plt.plot(adaptation['target_accuracy'], label="test acc. w/ ddc loss", marker='*', markersize=8)
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    # fig.savefig(os.path.join(pkldir, source + "_to_" + target + "_test_train_accuracies.jpg"))


def main():
    parser = argparse.ArgumentParser(description="average DeepCORAL accuracies")

    parser.add_argument("--source", default="amazon", type=str,
                        help="source data")

    parser.add_argument("--target", default="dslr", type=str,
                        help="target data")

    parser.add_argument("--no_epochs", default=100, type=int)

    args = parser.parse_args()

    get_classification_accuracy(source=args.source, target=args.target, no_epochs=args.no_epochs)


if __name__ == "__main__":
    main()
