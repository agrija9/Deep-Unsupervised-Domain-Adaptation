#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from tqdm import tnrange
from torch.autograd import Variable
from loss import CORAL_loss


def train(model, source_loader, target_loader,
          optimizer, epoch, lambda_factor, cuda=False):
    """
    This method fits the network params one epoch at a time.
    Implementation based on:
    https://github.com/SSARCandy/DeepCORAL/blob/master/main.py
    """

    model.train()

    results = [] # list to append loss values at each epoch

    # first cast into an iterable list the data loaders
    # data_source: (batch_size, channels, height, width)
    # data_target: (batch_size)
    # source[0][1][0].size() --> torch.Size([128, 3, 224, 224])

    # memory leakage
    source = list(enumerate(source_loader))
    target = list(enumerate(target_loader))
    train_steps = min(len(source), len(target))

    # start batch training
    for batch_idx in tnrange(train_steps):
        # fetch data in batches
        # _, source_data -> torch.Size([128, 3, 224, 224]), labels -> torch.Size([128])
        _, (source_data, source_label) = source[batch_idx]
        _, (target_data, _) = target[batch_idx] # unsupervised learning

        print("CUDA:", cuda)
        if cuda:
            # move to device
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()

        # create pytorch variables, the variables and functions build a dynamic graph of computation
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)

        # reset to zero optimizer gradients
        optimizer.zero_grad()

        # do a forward pass through network (recall DeepCORAL outputs source, target activation maps)
        output1, output2 = model(source_data, target_data)

        # compute losses (classification and coral loss)
        classification_loss = torch.nn.functional.cross_entropy(output1, source_label)
        coral_loss = CORAL_loss(output1, output2)

        print(type(coral_loss))

        # compute total loss (equation 6 paper)
        total_loss = classification_loss + lambda_factor*coral_loss

        # compute gradients of network (backprop in pytorch)
        total_loss.backward()

        # update weights of network
        optimizer.step()

        # append results for each batch iteration as dictionaries
        results.append({
            'epoch': epoch,
            'step': batch_idx + 1,
            'total_steps': train_steps,
            'lambda': lambda_factor,
            'coral_loss': coral_loss.item(), # coral_loss.data[0],
            'classification_loss': classification_loss.item(),  # classification_loss.data[0],
            'total_loss': total_loss.item() # total_loss.data[0]
        })

        # print training info
        print('Train Epoch: {:2d} [{:2d}/{:2d}]\t'
              'Lambda value: {:.4f}, Classification loss: {:.6f}, CORAL loss: {:.6f}, Total_Loss: {:.6f}'.format(
                  epoch,
                  batch_idx + 1,
                  train_steps,
                  lambda_factor,
                  classification_loss.item(), # classification_loss.data[0],
                  coral_loss.item(), # coral_loss.data[0],
                  total_loss.item() # total_loss.data[0]
              ))

    return results
