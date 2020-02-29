#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skimage import io
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms, datasets
# from torch.utils import model_zoo
import numpy as np
try: # import pytorch data getters
	from torch.hub import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url


"""
Created on Saturday Feb 22 2020

@authors: Alan Preciado, Santosh Muthireddy
"""
def load_pretrained_AlexNet(model):
# def alexnet(pretrained=False, progress=True, **kwargs):
    """
    AlexNet model architecture from the paper
    pretrained (bool): If True, returns a model pre-trained on ImageNet
    progress (bool): If True, displays a progress bar of the download to stderr
    """

    __all_ = ["AlexNet", "alexnet", "Alexnet"]

    model_url = {
        'alexnet':'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    }

    state_dict = load_state_dict_from_url(model_url['alexnet'], progress=progress)
	model_dict = model.state_dict() # check this one

	# filter out unmatching dictionary
	# reference: https://github.com/SSARCandy/DeepCORAL/blob/master/main.py
	state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

	model_dict.update(state_dict)
	model.load_state_dict(state_dict)


def show_image(dataset, domain, image_class, image_name):
	"""
	Plot images from given domain, class
	"""
	image_file = io.imread(os.path.join("data", dataset, domain, "images", image_class, image_name))
	plt.imshow(image_file)
	plt.pause(0.001)
	plt.figure()


def accuracy(prediction, target):
	pass


def save_model(model, path):
	torch.save(model.state_dict(), path)
	print("checkpoint saved in {}".format(path))


def load_model(model, path):
	model.load_state_dict(torch.load(path))
	print("checkpoint loaded from {}".format(path))


def get_mean_std_dataset(root_dir):
    """
    Function to compute mean and std of image dataset.
    Move batch_size param according to memory resources.
    retrieved from: https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
    """

    # data_domain = "amazon"
    # path_dataset = "datasets/office/%s/images" % data_domain

    transform = transforms.Compose([
            transforms.Resize((224, 224)), # original image size 300x300 pixels
            transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=root_dir,
                                   transform=transform)

    # set large batch size to get good approximate of mean, std of full dataset
    # batch_size: 4096, 2048
    data_loader = DataLoader(dataset, batch_size=2048,
                            shuffle=False, num_workers=0)

    mean = []
    std = []

    for i, data in enumerate(data_loader, 0):
        # shape is (batch_size, channels, height, width)
        npy_image = data[0].numpy()

        # compute mean, std per batch shape (3,) three channels
        batch_mean = np.mean(npy_image, axis=(0,2,3))
        batch_std = np.std(npy_image, axis=(0,2,3))

        mean.append(batch_mean)
        std.append(batch_std)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    mean = np.array(mean).mean(axis=0) # average over batch averages
    std = np.arry(std).mean(axis=0) # average over batch stds

    values = {
        "mean": mean,
        "std": std
    }

    return values
