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
from network import SoftmaxLoss

def download_dataset():
    train_dataset = torchvision.datasets.MNIST('./data', train=True,  download=False, 
        transform=transforms.ToTensor())
    test_dataset  = torchvision.datasets.MNIST('./data', train=False, download=False, 
        transform=transforms.ToTensor())
    return train_dataset,test_dataset


def main():
    train_dataset, test_dataset = download_dataset()

    model = SoftmaxLoss()
    params = torch.optim.Adam(model.parameters(),lr=0.001)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    loss_CE = nn.CrossEntropyLoss()

    n_epoch = 10
    # for i in range(n_epoch):
    #     loss_epoch = 0
    for picture,label in train_dataloader:
        out = model(picture)
        loss = loss_CE(out, label)
        loss.backward()
        params.step()
        # loss_epoch += loss.data
        print(f"/{n_epoch} train loss: {loss.data}")

    # ToDo 中間層の可視化
    ## 重み行列の特定の行成分の取得
    # model.Linear2.weight[0]
    # Question BackBornも更新する？







if __name__ == '__main__':
    main()