import math
import torch
import gpytorch
from matplotlib import pyplot as plt

def prepare_data(n):
    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x = torch.linspace(0, 1, n)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
    return train_x, train_y

def main():
    n = 100
    train_x, train_y = prepare_data(n)






if __name__ == '__main__':
    main()