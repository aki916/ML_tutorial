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
import torchvision.models as models

def download_dataset(transform):
    train_dataset = torchvision.datasets.MNIST('./data', train=True,  download=False, 
        transform=transform)
    test_dataset  = torchvision.datasets.MNIST('./data', train=False, download=False, 
        transform=transform)
    return train_dataset,test_dataset

def body_feature_model(model):
    """
    Returns a model that output flattened features directly from CNN body.
    """
    body, pool, head = list(model.children()) 
    return body, pool, head[:-1]

def main():
    transform = transforms.Compose([
        transforms.Resize((224,224), interpolation=2),
        transforms.Grayscale(num_output_channels=3) ,
        transforms.ToTensor()#PIL.Image.Image-> torch tensor
    ])

    train_dataset, test_dataset = download_dataset(transform)

    backbone_model = models.alexnet(pretrained=True)
    body, pool, head = body_feature_model(backbone_model)
    model = SoftmaxLoss(body, pool, head)
    params = torch.optim.Adam(model.parameters(),lr=0.001)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    loss_CE = nn.CrossEntropyLoss()

    n_epoch = 10
    # for i in range(n_epoch):
    #     loss_epoch = 0
    for picture,label in train_dataloader:
        print(picture.shape)
        out = model(picture)
        print(out.shape)
        exit()
        loss = loss_CE(out, label)
        loss.backward()
        params.step()
        # loss_epoch += loss.data
        print(f"/{n_epoch} train loss: {loss.data}")

    # ToDo 中間層の可視化
    ## 重み行列の特定の行成分の取得
    # model.Linear2.weight[0]

    # torch.save

    # validでハイパラを探索する部分
    # Datasetを自分で作った方が早いかも






if __name__ == '__main__':
    main()