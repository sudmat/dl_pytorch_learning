import torch
import os
import sys
import matplotlib.pyplot as plt

sys.path.append('D:/implement/dl/pytorch')

from layers.transformer.lr_scheduler import CosineWarmupScheduler

p = torch.nn.Parameter(torch.empty(4, 4))
optimizer = torch.optim.Adam(lr=1e-3, params=[p])

lr_scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=2000)

epochs = list(range(2000))
lrs = [lr_scheduler.get_lr_factor(e) for e in epochs]

plt.plot(epochs, lrs)
plt.show()
