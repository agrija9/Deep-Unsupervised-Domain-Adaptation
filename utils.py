#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Saturday Feb 22 2020

@authors: Alan Preciado, Santosh Muthireddy
"""

from skimage import io
import matplotlib.pyplot as plt
import os


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


# data getters
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

