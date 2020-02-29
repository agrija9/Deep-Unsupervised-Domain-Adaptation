#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Saturday Feb 22 2020

@authors: Alan Preciado, Santosh Muthireddy
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image


class OfficeAmazonDataset(Dataset):
    """Class to create an iterable dataset
    of images and corresponding labels """

    def __init__(self, image_folder_dataset, transform=None):
        super(OfficeAmazonDataset, self).__init__()
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform

    def __len__(self):
        return len(self.image_folder_dataset.imgs)

    def __getitem__(self, idx):
        # read image, class from folder_dataset given index
        img, img_label = image_folder_dataset[idx][0], image_folder_dataset[idx][1]

        # apply transformations (it already returns them as torch tensors)
        if self.transform is not None:
            self.transform(img)

        img_label_pair = {"image": img,
                         "class": img_label}

        return img_label_pair


def get_dataloader(dataset, batch_size, train_ratio=0.7):
    """
    Splits a dataset into train and test.
    Returns train_loader and test_loader.
    """

    def get_subset(indices, start, end):
        return indices[start:start+end]

    # Split train/val data ratios
    TRAIN_RATIO, VALIDATION_RATIO = train_ratio, 1-train_ratio
    train_set_size = int(len(dataset) * TRAIN_RATIO)
    validation_set_size = int(len(dataset) * VALIDATION_RATIO)

    # Generate random indices for train and val sets
    indices = torch.randperm(len(dataset))
    train_indices = get_subset(indices, 0, train_set_size)
    validation_indices = get_subset(indices, train_set_size, validation_set_size)
    # test_indices = get_subset(indices,train_count+validation_count,len(dataset))

    # Create sampler objects
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(validation_indices)

    # Create data loaders for data
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler, num_workers=4)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler, num_workers=4)

    return train_loader, val_loader
