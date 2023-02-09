import os
import sys

sys.path.append('D:/implement/dl/pytorch')

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.sequence_reverse import ReversedSequence
from layers.transformer.task1_reversed_seq import ReversedSeqTransformer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

pl.seed_everything(2023)

num_catoryies = 10
batch_size = 128

train_ds = ReversedSequence(
    num_categories=num_catoryies, seq_len=20, num_seqs=50000)
train_loader = DataLoader(
    dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

val_ds = ReversedSequence(num_categories=num_catoryies,
                          seq_len=20, num_seqs=1000)
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size,
                        shuffle=False, num_workers=0)

test_ds = ReversedSequence(
    num_categories=num_catoryies, seq_len=20, num_seqs=10000)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size,
                         shuffle=False, num_workers=0)

# model = ReversedSeqTransformer.load_from_checkpoint(r"experiments\transformer\models\seq_reverse\models\lightning_logs\version_0\checkpoints\epoch=2-step=1170.ckpt")
model = ReversedSeqTransformer.load_from_checkpoint(r"experiments\transformer\models\seq_reverse\models\lightning_logs\version_1\checkpoints\epoch=1-step=780.ckpt")

x, y = next(iter(test_loader))
x = F.one_hot(x, model.hparams.num_classes).float()

attn = model.get_attn_maps(x)
attn_avg = attn[0].mean(dim=(0, 1))

plt.imshow(attn_avg)

plt.show()

y_pred = model.forward(x)

print(1)

