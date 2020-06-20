import torch.nn as nn
from torch.nn import functional as F

class SoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.ConvTranspose2d(1, 3, 4, stride=1, padding=0, bias=False)
        self.Conv2 = nn.ConvTranspose2d(3, 3, 4, stride=1, padding=0, bias=False)
        self.Linear1 = nn.Linear(3468,2)
        self.Linear2 = nn.Linear(2,10)

    def forward(self, x):
        x = F.leaky_relu(self.Conv1(x),inplace=True)
        x = F.leaky_relu(self.Conv2(x),inplace=True)
        x = x.view(x.shape[0],-1)
        x = F.leaky_relu(self.Linear1(x),inplace=True)
        x = self.Linear2(x)

        return x