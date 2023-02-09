import os, sys
import matplotlib.pyplot as plt

sys.path.append('D:/implement/dl/pytorch')

from dataset.tiny_imagenet import TinyImagenet
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from layers.googlenet.GoogleNet import TinyGoogleNet
from torch.optim import Adam
from torch import nn
import torch

class TinyGoogleNetModule(pl.LightningModule):

    def __init__(self, lr=10e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = TinyGoogleNet(in_channels=3, out_channels=200)
        self.loss_module = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros((1, 3, 56, 56), dtype=torch.float32)
    
    # used for non-training purpose
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return [optimizer]
    
    def training_step(self, batch, batch_index):
        x, gt, fn = batch
        y0, y1, y2 = self.model(x)
        acc = (y0.argmax(dim=-1) == gt).float().mean()

        loss0 = self.loss_module(y0, gt)
        loss1 = self.loss_module(y1, gt)
        loss2 = self.loss_module(y2, gt)

        loss = loss0 + 0.3 * loss1 + 0.3 * loss2
        
        self.log('train acc', acc, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.log('train loss0', loss0, batch_size=x.shape[0])
        self.log('train loss1', loss1, batch_size=x.shape[0])
        self.log('train loss2', loss2, batch_size=x.shape[0])

        return loss
    
    def validation_step(self, batch, batch_index):
        x, gt, fn = batch
        y0, y1, y2 = self.model(x)
        acc = (y0.argmax(dim=-1) == gt).float().mean()
        self.log('val_acc', acc, batch_size=x.shape[0])

    def test_step(self, batch, batch_index):
        x, gt, fn = batch
        y0, y1, y2 = self.model(x)
        acc = (y0.argmax(dim=-1) == gt).float().mean()
        self.log('val acc', acc, batch_size=x.shape[0])

if __name__ == '__main__':

    train_transform = transforms.Compose([
        transforms.Resize((56, 56)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((56, 56), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor()
    ])
    train_ds = TinyImagenet('D:/implement/dl/data/tiny-imagenet-200', split='train', transforms=train_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

    val_transform = transforms.Compose([
        transforms.Resize((56, 56)), 
        transforms.ToTensor()
    ])
    val_ds = TinyImagenet('D:/implement/dl/data/tiny-imagenet-200', split='val', transforms=val_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    debug_ds = TinyImagenet('D:/implement/dl/data/tiny-imagenet-200', split='debug', transforms=train_transform)
    debug_loader = DataLoader(debug_ds, batch_size=1, shuffle=False, num_workers=0)

    trainer = pl.Trainer(default_root_dir='experiments/googlenet/models/',
                         accelerator="gpu",                     
                         devices=1,
                         max_epochs=2,
                         callbacks=[pl.callbacks.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         enable_progress_bar=True)       
    
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    pl.seed_everything(42)
    model = TinyGoogleNetModule()
    trainer.fit(model, train_loader, val_loader)