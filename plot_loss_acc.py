#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path


"""
Created on Friday Mar 13 2020

@authors: Alan Preciado, Santosh Muthireddy
"""
def plot_loss_acc(source, target, no_epochs):

    # specify path where in log folder where training logs are saved
    path = Path("/logs/" + source + "_to_" + target + "/" + no_epochs)

    if path.is_dir():
        print("path to pkl log files:", path)

    else:
        print("folder with pkl data does not exist, must trrain model")
        return None

    # load dictionaries with log information
    path_adapt_log = [path + "/adaptation_training_statistic.pkl",
                      path + "/adaptation_testing_s_statistic.pkl",
                      path + "/adaptation_testing_t_statistic.pkl"]

    path_no_adapt_log = [path + "/no_adaptation_training_statistic.pkl",
                         path + "/no_adaptation_training_statistic.pkl",
                         path + "/no_adaptation_training_statistic.pkl"]

    adapt_training_dict = pickle.load(open(path_adapt_log[0], 'rb'))
    adapt_testing_source_dict = pickle.load(open(path_adapt_log[1], 'rb'))
    adapt_testing_target_dict = pickle.load(open(path_adapt_log[2], 'rb'))

    no_adapt_training_dict = pickle.load(open(path_no_adapt_log[0], 'rb'))
    no_adapt_testing_source_dict = pickle.load(open(path_no_adapt_log[1], 'rb'))
    no_adapt_testing_target_dict = pickle.load(open(path_no_adapt_log[2], 'rb'))

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

    plt.xlabel("epochs", fontsize=15)
    plt.ylabel("classification accuracy (%)", fontsize=15)

    plt.plot(adaptation['target_accuracy'], label="test acc. w/ coral loss", marker='*', markersize=8)
    plt.plot(no_adaptation['target_accuracy'], label="test acc. w/o coral loss", marker='.', markersize=8)

    plt.plot(adaptation['source_accuracy'], label="training acc. w/ coral loss", marker='^', markersize=8)
    plt.plot(no_adaptation['source_accuracy'], label="training acc. w/o coral loss", marker='+', markersize=8)

    plt.legend(loc="best")
    plt.grid()
    plt.show()
    fig.savefig(path + "/webcam_to_amazon_test_train_accuracies.jpg")

    # plot accuracies for test data in source and target domains
    fig=plt.figure(figsize=(8, 6), dpi=100)
    fig.show()

    plt.xlabel("epochs", fontsize=15)
    plt.ylabel("classification accuracy (%)", fontsize=15)

    plt.plot(adaptation["classification_loss"], label="classification_loss", marker='*', markersize=8)
    plt.plot(adaptation["coral_loss"], label="coral_loss", marker='.', markersize=8)

    plt.legend(loc="best")
    plt.grid()
    plt.show()
    fig.savefig(path + "/webcam_to_amazon_train_losses.jpg")
