import torch.nn.functional as F
from .transfomer_predictor import TransformerPredictor
import torch

class AnomalyDetectionTransformer(TransformerPredictor):

    def _calc_loss(self, batch, mode):
        indices, x,  labels = batch
        # y = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=torch.int64)
        # y[:, -1] = 1
        y = torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.int64) + 9
        # p = torch.zeros(indices.shape[0], device=x.device, dtype=torch.int32) + indices.shape[1]
        # y = F.one_hot(p, self.hparams.num_classes).float()
        y_pred = self.forward(x, add_positional_encoding=False)
        print(y_pred[:, -1])

        loss = F.cross_entropy(y_pred.squeeze(-1), y.view(-1))
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