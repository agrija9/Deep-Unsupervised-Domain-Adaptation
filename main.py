#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#TODO: arparsing, save model, plot curves
# TODO: try different combinations between 3 datasets to test adaptation

# main.py
from __future__ import division
import argparse
import warnings
from tqdm import tnrange
warnings.filterwarnings("ignore")

import torch
from torch.autograd import Variable

from loss import CORAL_loss
from utils import load_pretrained_AlexNet, save_model, load_model
from dataloader import get_office_dataloader
from model import DeepCORAL, AlexNet

# define train parameters as in the paper (page 5)
CUDA = True if torch.cuda.is_available() else False
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
BATCH_SIZE = [32, 32] # batch_s, batch_t [128, 56]
EPOCHS = 1

# create dataloaders (Amazon as source and Webcam as target)
source_loader = get_office_dataloader(name_dataset="amazon", batch_size=BATCH_SIZE[0])
target_loader = get_office_dataloader(name_dataset="webcam", batch_size=BATCH_SIZE[1])

# argparse ...
model = DeepCORAL(num_classes=31) # input no. of classes in custom dataset

# define optimizer pytorch: https://pytorch.org/docs/stable/optim.html
# specify per-layer learning rates: 10*learning_rate for last two fc layers according to paper
optimizer = torch.optim.SGD([
    {"params": model.sharedNetwork.parameters()},
    {"params": model.fc8.parameters(), "lr":10*LEARNING_RATE},
], lr=LEARNING_RATE, momentum=MOMENTUM)

if CUDA:
    model = model.cuda()
    print("using cuda...")

# load pre-trained model --> TODO: check it is loading properly
# if args.load is not NONE:
    # load_model(model, args.load)
# else:
    # load_pretrained_AlexNet(model.sharedNetwork, progress=True)

load_pretrained_AlexNet(model.sharedNetwork, progress=True)

# store statistics of train/test
training_s_statistic = []
testing_s_statistic = []
testing_t_statistic = []

# iterate over epochs
for epoch in tnrange(EPOCHS):
    # compute lambda value
    _lambda = (epoch+1)/EPOCHS
    
    result_train = train(model, optimizer, epoch+1, _lambda)
    
    print('###EPOCH {}: Classification: {:.6f}, CORAL loss: {:.6f}, Total_Loss: {:.6f}'.format(
            epoch+1,
            sum(row['classification_loss'] / row['total_steps'] for row in result_train),
            sum(row['coral_loss'] / row['total_steps'] for row in result_train),
            sum(row['total_loss'] / row['total_steps'] for row in result_train),
        ))
    
    training_s_statistic.append(result_train)
    
    # perform testing simultaneously: classification accuracy on both dataset
    test_source = test(model, source_loader, epoch)
    test_target = test(model, target_loader, epoch)
    testing_s_statistic.append(test_source)
    testing_t_statistic.append(test_target)
    
    print('###Test Source: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            epoch+1,
            test_source['average_loss'],
            test_source['correct_class'],
            test_source['total_elems'],
            test_source['accuracy %'],
        ))

    print('###Test Target: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            epoch+1,
            test_target['average_loss'],
            test_target['correct_class'],
            test_target['total_elems'],
            test_target['accuracy %'],
    ))