#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Saturday Feb 25 2020

@authors: Alan Preciado, Santosh Muthireddy
"""
import torch 


def CORAL_loss(source, target):
	"""
	From the paper, the vectors that compose Ds and Dt are D-dimensional vectors.
	:param source: torch tensor: source data (Ds) with dimensions DxNs
	:param target: torch tensor: target data (Dt) with dimensons DxNt
	"""

	d = source.size(1) # d-dimensional vectors (same for source, target)

	source_covariance = compute_covariance(source)
	target_covariance = compute_covariance(target)
	
	# take Frobenius norm (https://pytorch.org/docs/stable/torch.html)
	loss = torch.norm(torch.mul((source_covariance-target_covariance),(source_covariance-target_covariance)), p="fro")
	# loss = torch.norm(torch.mm((source_covariance-target_covariance),(source_covariance-target_covariance)), p="fro")
	loss = loss / (4*d*d)
	
	return loss


def compute_covariance(data):
	"""
	Compute covariance matrix for given dataset as shown in paper.
	Equations 2 and 3.
	:param data: torch tensor: input source/target data
	"""

	n = data.size(0) # data dimensions: nxd (this is Ns or Nt)

	ones_vector = torch.ones(n).resize(1, n) # 1xN dimensional vector


	# proper matrix multiplication

