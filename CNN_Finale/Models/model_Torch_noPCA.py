import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


def torch_noPCA(width, height):
    class Reshape(nn.Module):
        def forward(self, x):
            return x.view(-1, 1, width, height)

    model = nn.Sequential(
        Reshape(),
        nn.Conv2d(1, 10, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(10),
        nn.Conv2d(10, 10, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(10*8*8, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    return model
    
    
