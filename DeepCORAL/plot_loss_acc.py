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


"""
Created on Friday Mar 13 2020
Modified on Saturday Mar 21 2020
@authors: Alan Preciado, Santosh Muthireddy
"""
def plot_loss_acc(source, target, no_epochs):
    plt.rcParams.update({'font.size': 22})

    # specify path where in log folder where training logs are saved
    pkldir = os.path.join("logs", source + "_to_" + target,
                          str(no_epochs) + "_epochs_128_s_128_t_batch_size")

    if Path(pkldir).is_dir():
        print("directory with pkl files is:", pkldir)

    else:
        print("folder with pkl data does not exist, must train model")
        return None

    # load dictionaries with log information
    path_adapt_log = [pkldir + "/adaptation_training_statistic.pkl",
                      pkldir + "/adaptation_testing_s_statistic.pkl",
                      pkldir + "/adaptation_testing_t_statistic.pkl"]

    path_no_adapt_log = [pkldir + "/no_adaptation_training_statistic.pkl",
                         pkldir + "/no_adaptation_testing_s_statistic.pkl",
                         pkldir + "/no_adaptation_testing_t_statistic.pkl"]

    print(">>>Loading pkl files<<<")
    adapt_training_dict = pickle.load(open(path_adapt_log[0], 'rb'))
    adapt_testing_source_dict = pickle.load(open(path_adapt_log[1], 'rb'))
    adapt_testing_target_dict = pickle.load(open(path_adapt_log[2], 'rb'))

    no_adapt_training_dict = pickle.load(open(path_no_adapt_log[0], 'rb'))
    no_adapt_testing_source_dict = pickle.load(open(path_no_adapt_log[1], 'rb'))
    no_adapt_testing_target_dict = pickle.load(open(path_no_adapt_log[2], 'rb'))
    print(">>>pkl files loaded correctly<<<")

    print(np.shape(adapt_testing_source_dict))
    print(np.shape(no_adapt_testing_source_dict))

    # create dictionary structures for adaptation and no-adaptation results

    # (w coral loss)
    adaptation = {
        "classification_loss": [],
        "coral_loss": [],
        "source_accuracy": [],
        "target_accuracy": []
    }

    # (w/o coral loss)
    no_adaptation = {
        "source_accuracy": [],
        "target_accuracy": []
    }

    # get average coral and classification loss for steps in each epoch
    # get accuracy obtained in each epoch
    for epoch_idx in range(len(adapt_training_dict)): # epoch
        coral_loss = 0
        class_loss = 0

        for step_idx in range(len(adapt_training_dict[epoch_idx])):
            coral_loss += adapt_training_dict[epoch_idx][step_idx]["coral_loss"]
            class_loss += adapt_training_dict[epoch_idx][step_idx]["classification_loss"]

        # store average losses in general adaptation dictionary
        adaptation["classification_loss"].append(class_loss/len(adapt_training_dict[epoch_idx]))
        adaptation["coral_loss"].append(coral_loss/len(adapt_training_dict[epoch_idx]))
        adaptation["source_accuracy"].append(adapt_testing_source_dict[epoch_idx]["accuracy %"])
        adaptation["target_accuracy"].append(adapt_testing_target_dict[epoch_idx]["accuracy %"])

        # store accuracies in no-adaptation dictionary
        no_adaptation["source_accuracy"].append(no_adapt_testing_source_dict[epoch_idx]["accuracy %"])
        no_adaptation["target_accuracy"].append(no_adapt_testing_target_dict[epoch_idx]["accuracy %"])

    # plot accuracies for test data in source and target domains
    fig=plt.figure(figsize=(8, 6), dpi=100)
    fig.show()

    plt.xlabel("epochs", fontsize=25)
    plt.ylabel("classification accuracy (%)", fontsize=25)

    plt.plot(adaptation['target_accuracy'], label="test acc. w/ coral loss", marker='*', markersize=14)
    plt.plot(no_adaptation['target_accuracy'], label="test acc. w/o coral loss", marker='.', markersize=14)

    plt.plot(adaptation['source_accuracy'], label="training acc. w/ coral loss", marker='^', markersize=14)
    plt.plot(no_adaptation['source_accuracy'], label="training acc. w/o coral loss", marker='+', markersize=14)

    plt.legend(loc="best")
    plt.grid()
    plt.show()
    fig.suptitle(source + "_to_" + target)
    fig.savefig(os.path.join(pkldir, source + "_to_" + target + "_test_train_accuracies.jpg"))

    # plot losses for test data in source and target domains
    fig=plt.figure(figsize=(8, 6), dpi=100)
    fig.show()

    plt.xlabel("epochs", fontsize=25)
    plt.ylabel("loss", fontsize=25)

    plt.plot(adaptation["classification_loss"], label="classification_loss", marker='*', markersize=14)
    plt.plot(adaptation["coral_loss"], label="coral_loss", marker='.', markersize=14)

    plt.legend(loc="best")
    plt.grid()
    plt.show()
    fig.suptitle(source + "_to_" + target)
    fig.savefig(os.path.join(pkldir, source + "_to_" + target + "_train_losses.jpg"))


def main():
    parser = argparse.ArgumentParser(description="plots DeepCORAL")

    parser.add_argument("--source", default="amazon", type=str,
                        help="source data")

    parser.add_argument("--target", default="dslr", type=str,
                        help="target data")

    parser.add_argument("--no_epochs", default=100, type=int)

    args = parser.parse_args()

    plot_loss_acc(source=args.source, target=args.target, no_epochs=args.no_epochs)


if __name__ == "__main__":
    main()
