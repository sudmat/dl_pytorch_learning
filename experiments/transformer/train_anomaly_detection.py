import os
import sys

sys.path.append('D:/implement/dl/pytorch')

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.cifar100 import SetAnomalyDataset
from layers.transformer.task2_anomaly_detection import AnomalyDetectionTransformer
import torch

pl.seed_everything(2023)

set_size = 10
batch_size = 32

root = 'D:/implement/dl/data/cifar-100'

train_ds = SetAnomalyDataset(root, 'train', set_size=set_size)
train_loader = DataLoader(
    dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

val_ds = SetAnomalyDataset(root, 'val', set_size=set_size)
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size,
                        shuffle=False, num_workers=0)

test_ds = SetAnomalyDataset(root, 'test', set_size=set_size)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size,
                         shuffle=False, num_workers=0)

trainer = pl.Trainer(default_root_dir='experiments/transformer/models/anomaly_detection/models/',
                     accelerator="gpu",
                     devices=1,
                     max_epochs=10,
                     callbacks=[pl.callbacks.ModelCheckpoint(
                         save_weights_only=True, mode="max", monitor="val_acc")],
                     enable_progress_bar=True)

model = AnomalyDetectionTransformer(
    in_dim=512,
    num_classes=1,
    model_dim=512,
    num_blocks=1,
    num_heads=1,
    linear_dim=1024,
    lr=5e-4,
    warmup=1,
    max_iters=trainer.max_epochs * len(train_loader),
    dropout=0,
    input_dropout=0
)

trainer.fit(model, train_loader, val_loader)

test_result = trainer.test(model, test_loader, verbose=False)

print(f"Test accuracy: {(100.0 * test_result[0]['test_acc']):4.2f}%")
# trainer.logger._log_graph = True
# trainer.logger._default_hp_metric = None