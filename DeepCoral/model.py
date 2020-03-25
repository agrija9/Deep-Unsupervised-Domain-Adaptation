#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np


class DeepCORAL(nn.Module):
	"""
	DeepCORAL network as defined in the paper.
	Network architecture based on following repository:
    https://github.com/SSARCandy/DeepCORAL/blob/master/models.py
    :param num_classes: int --> office dataset has 31 different classes
	"""
	def __init__(self, num_classes=1000):
		super(DeepCORAL, self).__init__()
		self.sharedNetwork = AlexNet()
		self.fc8 = nn.Linear(4096, num_classes) # fc8 activation

		# initiliaze fc8 weights according to the CORAL paper (N(0, 0.005))
		self.fc8.weight.data.normal_(0.0, 0.005)

	def forward(self, source, target): # computes activations for BOTH domains
		source = self.sharedNetwork(source)
		source = self.fc8(source)

		target = self.sharedNetwork(target)
		target = self.fc8(target)

		return source, target



class AlexNet(nn.Module):
	"""
	AlexNet model obtained from official Pytorch repository:
    https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
	"""
	def __init__(self, num_classes=1000):
		super(AlexNet, self).__init__()

		self.features = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)

		self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6,6))

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True), # take fc8 (without activation)
			# nn.Linear(4096, num_classes),
			)

	def forward(self, x):
		# define forward pass of network
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1) # flatten to input into classifier
		# x = x.view(x.size(0), 246 * 6 * 6)
		x = self.classifier(x)

		return x
