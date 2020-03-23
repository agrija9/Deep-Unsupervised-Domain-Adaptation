#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np


def init_weights(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
		nn.init.kaiming_uniform_(m.weight)
		nn.init.zeros_(m.bias)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight, 1.0, 0.02)
		nn.init.zeros_(m.bias)
	elif classname.find('Linear') != -1:
		nn.init.xavier_normal_(m.weight)
		nn.init.zeros_(m.bias)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
	return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
	def fun1(grad):
		return -coeff * grad.clone()

	return fun1

class baseNetwork(nn.Module):
	"""
	DeepCORAL network as defined in the paper.
	Network architecture based on following repository:
    https://github.com/SSARCandy/DeepCORAL/blob/master/models.py
    :param num_classes: int --> office dataset has 31 different classes
	"""
	def __init__(self, num_classes=1000,bottleneck_dim=256):
		super(baseNetwork, self).__init__()
		self.sharedNetwork = AlexNet()
		self.bottleneck = nn.Linear(4096,bottleneck_dim)
		self.fc8 = nn.Linear(bottleneck_dim, num_classes) # fc8 activation

		# initiliaze fc8 weights according to the CORAL paper (N(0, 0.005))
		self.fc8.weight.data.normal_(0.0, 0.005)

	def forward(self, source): # computes activations for BOTH domains
		features = self.sharedNetwork(source)
		features = self.bottleneck(features)
		outputs = self.fc8(features)


		return features, outputs



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

class AdversarialNetwork(nn.Module):
	"""
    AdversarialNetwork obtained from official CDAN repository:
    https://github.com/thuml/CDAN/blob/master/pytorch/network.py
    """
	def __init__(self, in_feature, hidden_size):
		super(AdversarialNetwork, self).__init__()
		# print(hidden_size)
		self.ad_layer1 = nn.Linear(in_feature, hidden_size)
		self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
		self.ad_layer3 = nn.Linear(hidden_size, 1)
		self.relu1 = nn.ReLU()
		self.relu2 = nn.ReLU()
		self.dropout1 = nn.Dropout(0.5)
		self.dropout2 = nn.Dropout(0.5)
		self.sigmoid = nn.Sigmoid()
		self.apply(init_weights)
		self.iter_num = 0
		self.alpha = 10
		self.low = 0.0
		self.high = 1.0
		self.max_iter = 10000.0

	def forward(self, x):
		print("inside ad net forward",self.training)
		print(x.size())
		if self.training:
			self.iter_num += 1
		coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
		x = x * 1.0
		x.register_hook(grl_hook(coeff))
		x = self.ad_layer1(x)
		x = self.relu1(x)
		x = self.dropout1(x)
		x = self.ad_layer2(x)
		x = self.relu2(x)
		x = self.dropout2(x)
		y = self.ad_layer3(x)
		y = self.sigmoid(y)
		return y


	def output_num(self):
		return 1

	def get_parameters(self):
		return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]
