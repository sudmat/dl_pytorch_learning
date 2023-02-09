import os
import sys

sys.path.append('D:/implement/dl/pytorch')

import torch
from layers.transformer.multihead_attention import MultiheadAttention, MultiheadAttentionGT

x = torch.rand((4, 100, 64))

layer_gt = MultiheadAttentionGT(input_dim=64, embed_dim=128, num_heads=8)
y_gt = layer_gt(x)

layer = MultiheadAttention(in_dim=64, out_dim=128, num_head=8)
layer.same_as_gt(layer_gt)
y = layer(x)

assert y.mean() == y_gt.mean()

print(1)
