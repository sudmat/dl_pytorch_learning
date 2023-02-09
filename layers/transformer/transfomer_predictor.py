from torch import nn
import torch
import pytorch_lightning as pl
from .transformer_encoder import Encoder, PE
from torch import optim
from .lr_scheduler import CosineWarmupScheduler

class TransformerPredictor(pl.LightningModule):

    def __init__(self, in_dim, num_classes, model_dim, num_blocks, num_heads, 
    linear_dim, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):

        self.input_net = nn.Sequential(
            nn.Dropout(p=self.hparams.input_dropout),
            nn.Linear(self.hparams.in_dim, self.hparams.model_dim)
        )

        self.transformer = Encoder(
            in_dim=self.hparams.model_dim, 
            hidden_fw_dim=self.hparams.linear_dim, 
            num_heads=self.hparams.num_heads, 
            num_blocks=self.hparams.num_blocks
            )

        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        )

        self.pe = PE(500, self.hparams.model_dim)


    def forward(self, x, mask=None, add_positional_encoding=True):

        x = self.input_net(x)

        if add_positional_encoding:
            x = self.pe(x)

        x = self.transformer(x, mask)

        x = self.output_net(x)

        return x

    @torch.no_grad()
    def get_attn_maps(self, x, mask=None, add_positional_encoding=True):
        
        x = self.input_net(x)

        if add_positional_encoding:
            x = self.pe(x)

        atten_maps = self.transformer.get_attn_maps(x, mask)
        return atten_maps
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.hparams.lr)
        lr_scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup, self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError