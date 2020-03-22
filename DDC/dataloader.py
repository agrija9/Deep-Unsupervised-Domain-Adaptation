#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import os
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from utils import get_mean_std_dataset


"""
Created on Saturday Feb 22 2020

@authors: Alan Preciado, Santosh Muthireddy
"""

def get_office_dataloader(name_dataset, batch_size, train=True):
    """
    Creates dataloader for the datasets in office datasetself.
    Uses get_mean_std_dataset() to compute mean and std along the
    color channels for the datasets in office.
    """

    # root dir (local pc or colab)
    root_dir = "/content/office/%s/images" % name_dataset
    # root_dir = "datasets/office/%s/images" % name_dataset
    # root_dir = "/content/drive/My Drive/office/%s/images" % name_dataset

    __datasets__ = ["amazon", "dslr", "webcam"]

    if name_dataset not in __datasets__:
        raise ValueError("must introduce one of the three datasets in office")

    # Ideally compute mean and std with get_mean_std_dataset.py
    # https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/master/data_loader.py
    mean_std = {
        "amazon":{
            "mean":[0.7923, 0.7862, 0.7841],
            "std":[0.3149, 0.3174, 0.3193]
        },
        "dslr":{
            "mean":[0.4708, 0.4486, 0.4063],
            "std":[0.2039, 0.1920, 0.1996]
        },
        "webcam":{
            "mean":[0.6119, 0.6187, 0.6173],
            "std":[0.2506, 0.2555, 0.2577]
        }
    }

    # compose image transformations
    data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[name_dataset]["mean"],
                                 std=mean_std[name_dataset]["std"])
        ])

    # retrieve dataset using ImageFolder
    # datasets.ImageFolder() command expects our data to be organized
    # in the following way: root/label/picture.png
    dataset = datasets.ImageFolder(root=root_dir,
                                   transform=data_transforms)

    # Dataloader is able to spit out random samples of our data,
    # so our model won’t have to deal with the entire dataset every time.
    # shuffle data when training
    dataset_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=train,
                                num_workers=4,
                                drop_last=True)

    return dataset_loader
