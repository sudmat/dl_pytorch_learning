import torch.nn.functional as F
from .transfomer_predictor import TransformerPredictor
import torch

class ReversedSeqTransformer(TransformerPredictor):

    def _calc_loss(self, batch, mode):
        x, y = batch
        x = F.one_hot(x, self.hparams.num_classes).float()

        y_pred = self.forward(x, add_positional_encoding=True)

        loss = F.cross_entropy(y_pred.view(-1,y_pred.size(-1)), y.view(-1))
        acc = (y_pred.argmax(dim=-1) == y).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, mode='train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        self._calc_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calc_loss(batch, mode='test')