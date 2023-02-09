import os, sys

sys.path.append('D:/implement/dl/pytorch')

from layers.googlenet.GoogleNet import GoogleNet
import torch
from torch import nn

x = torch.rand((1, 3, 224, 224))

net = GoogleNet()
y = net(x)

print(sum([item.numel() for item in net.parameters()]))

print(1)
