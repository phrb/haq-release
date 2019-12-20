#! /usr/bin/python3

import os
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = datasets.ImageFolder("data/train", train_transform)
