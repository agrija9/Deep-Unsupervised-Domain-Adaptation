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
    validation_indices = get_subset(indices,train_set_size,validation_set_size)
    # test_indices = get_subset(indices,train_count+validation_count,len(dataset))

    # Create sampler objects
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(validation_indices)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler, num_workers=0)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler, num_workers=0)

    return train_loader, val_loader


def get_office_dataloader(name_dataset, batch_size, train=True):
    """
    Creates dataloader for the datasets in office datasetself.
    Uses get_mean_std_dataset() to compute mean and std along the
    color channels for the datasets in office.
    """

    # root dir (local pc or colab)
    root_dir = "office/%s/images" % name_dataset
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
                                num_workers=0,drop_last=True)

    return dataset_loader
