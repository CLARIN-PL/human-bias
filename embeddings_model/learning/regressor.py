import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pytorch_lightning as pl


class Regressor(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        output = self.forward(x)

        loss = nn.MSELoss()(output.float(), y.float())

        self.log('train_loss',  loss, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outs):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        output = self.forward(x)

        loss = nn.MSELoss()(output, y)

        self.log('valid_loss', loss, prog_bar=True)
        
        return loss

    def validation_epoch_end(self, outs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer