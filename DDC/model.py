#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class DDCNet(nn.Module):
	"""
	Deep domain confusion network as defined in the paper:
	https://arxiv.org/abs/1412.3474
    :param num_classes: int --> office dataset has 31 different classes
	"""
	def __init__(self, num_classes=1000):
		super(DDCNetwork, self).__init__()
		self.sharedNetwork = AlexNet()

		self.bottleneck = nn.Sequential(
			nn.Linear(4096, 256),
			nn.ReLU(inplace=True)
		)

		# fc8 activation (final_classifier)
		# self.fc8 = nn.Linear(4096, num_classes)
		self.fc8 = nn.Sequential(
			nn.Linear(256, num_classes)
		)

		self.fc8.weight.data.normal_(0.0, 0.005)

	def forward(self, source, target): # computes activations for BOTH domains
		source = self.sharedNetwork(source)
		source = self.bottleneck(source)
		source = self.fc8(source)

		target = self.sharedNetwork(target)
		target = self.bottleneck(target)
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
