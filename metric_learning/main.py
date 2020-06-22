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
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import datetime
from torch.autograd import grad as torch_grad
from torchvision.transforms.functional import to_pil_image, to_tensor
from network import *
import torchvision.models as models
from torch.nn import functional as F


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

def visualize_feature_center(test_dataloader,BB,metric_fc):
    plt.figure()
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    cmap = plt.get_cmap("tab10")
    count = 10
    c = 0

    # データの特徴ベクトルの可視化
    with torch.no_grad():
        for picture, label in test_dataloader:
            feature = BB(picture)
            normalized_feature = F.normalize(feature)
            sc = plt.scatter(normalized_feature[:,0],normalized_feature[:,1],c=label,cmap=cmap,alpha=0.8)
            # print()
            # break
            c += 1
            if c == count:
                break

    # クラスタの中心の可視化
    normalized_center = F.normalize(metric_fc.weight).detach().numpy()
    for i in range(10):
        plt.plot([0,normalized_center[i,0]],[0,normalized_center[i,1]],c=cmap(i))
    # sc = plt.scatter(normalized_center[:,0],normalized_center[:,1],c=label,cmap=cmap)

    # 2dimならそのまま可視化、それ以上なら次元圧縮

    plt.colorbar(sc)
    plt.show()

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def train(train_dataloader,metric_fc, BB, params, n_epoch):
    # # for i in range(n_epoch):
    loss_epoch = 0
    count = 30
    c = 0
    loss_CE = nn.CrossEntropyLoss()

    for picture,label in train_dataloader:
        feature = BB(picture)
        output = metric_fc(feature, label)
        print(output)
        loss = loss_CE(output, label)
        loss.backward()
        params.step()
        loss_epoch += loss.data
        c += 1
        if c == count:
            break
        print(f"/{n_epoch} train loss: {loss.data}")

    return metric_fc, BB, params

def main():
    transform = transforms.Compose([
        transforms.Resize((224,224), interpolation=2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),#PIL.Image.Image-> torch tensor,
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset, test_dataset = download_dataset(transform)

    backbone_model = models.alexnet(pretrained=True)
    body, pool, head = body_feature_model(backbone_model)
    # BB = resnet34()
    feature_dim = 1000
    BB = BackBone(body, pool, head, feature_dim)
    metric_fc = SoftmaxLoss(in_features=feature_dim,out_features=10)
    # metric_fc = ArcMarginProduct(in_features=feature_dim,out_features=10)
    params = torch.optim.Adam([{'params': BB.parameters()}, {'params': metric_fc.parameters()}],lr=0.001)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64, shuffle=True, num_workers=1)

    n_epoch = 10

    metric_fc, BB, params = train(train_dataloader,metric_fc, BB, params, n_epoch)
    exit()

    # with no grad
    visualize_feature_center(test_dataloader,BB,metric_fc)

    save_path = './'
    name = 'SoftmaxLoss'
    iter_cnt = '1'
    # save_model(model, save_path, name, iter_cnt)

    # validでハイパラを探索する部分
    # 特定のクラスを除外することを考えるとDatasetを自分で作った方が早いかも


if __name__ == '__main__':
    main()