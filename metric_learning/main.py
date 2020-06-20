from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import torch.nn as nn
import tqdm
import statistics
import time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import datetime
from torch.autograd import grad as torch_grad
from torchvision.transforms.functional import to_pil_image, to_tensor


def download_dataset():
    train_dataset = torchvision.datasets.MNIST('./data', train=True,  download=True, 
        transform=transforms.ToTensor())
    test_dataset  = torchvision.datasets.MNIST('./data', train=False, download=True, 
        transform=transforms.ToTensor())
    return train_dataset,test_dataset


def main():
    train_dataset, test_dataset = download_dataset()
    print(train_dataset)










if __name__ == '__main__':
    main()