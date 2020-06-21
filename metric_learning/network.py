import torch.nn as nn
from torch.nn import functional as F
import torch

class SoftmaxLoss(nn.Module):
    def __init__(self, body, pool, head):
        super().__init__()
        self.body = body
        self.pool = pool
        self.head = head
        self.Linear1 = nn.Linear(4096,100)
        self.Linear2 = nn.Linear(100,10)

    def forward(self, x):
        x = self.body(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = F.leaky_relu(self.Linear1(x),inplace=True)
        x = self.Linear2(x)

        return x