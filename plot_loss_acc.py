#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle


"""
Created on Friday Mar 13 2020

@authors: Alan Preciado, Santosh Muthireddy
"""
def plot_loss_acc():


    # load dictionaries with log information
    path_adapt_log = ["/content/adaptation_training_statistic.pkl",
                      "/content/adaptation_testing_s_statistic.pkl",
                      "/content/adaptation_testing_t_statistic.pkl"]

    path_no_adapt_log = ["/content/no_adaptation_training_statistic.pkl",
                         "/content/no_adaptation_training_statistic.pkl",
                         "/content/no_adaptation_training_statistic.pkl"]
                        
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
