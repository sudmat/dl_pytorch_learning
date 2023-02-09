import os, sys

sys.path.append('D:/implement/dl/pytorch')

from layers.googlenet.inception_block import InceptionBlock, InceptionBlockV1
import torch
from torch import nn

x = torch.rand((1, 3, 128, 128))

block = InceptionBlock(c_in=3, c_out_1x1=64, c_out_3x3_dr=96, c_out_3x3=128, c_out_5x5_dr=16, c_out_5x5=32, c_out_mp=32, act_fn=nn.ReLU)
y = block(x)

block1 = InceptionBlockV1(c_in=3, c_red={'3x3': 96, '5x5': 16}, c_out={'1x1': 64, '3x3': 128, '5x5': 32, 'max': 32}, act_fn=nn.ReLU)
y1 = block1(x)

print(y.shape, y1.shape)
print([item.numel() for item in block.parameters()])
print([item.numel() for item in block1.parameters()])
