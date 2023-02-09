import os
import sys

sys.path.append('D:/implement/dl/pytorch')

import torch
from layers.transformer.transformer_encoder import Encoder


x = torch.randn((4, 100, 32))

encoder = Encoder(in_dim=x.shape[-1], hidden_fw_dim=64, num_heads=4, num_blocks=6)

y = encoder(x)

print(1)