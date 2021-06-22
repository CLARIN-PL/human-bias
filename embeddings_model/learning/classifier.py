import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pytorch_lightning as pl


class Classifier(pl.LightningModule):
    def __init__(self, model, class_dims, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.class_dims = class_dims

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)

        loss = 0
        class_dims = self.class_dims
        for cls_idx in range(len(class_dims)):
            start_idx =  sum(class_dims[:cls_idx])
            end_idx =  start_idx + class_dims[cls_idx]

            if y.ndim > 1:
                loss = loss + nn.CrossEntropyLoss()(output[:, start_idx:end_idx], y[:, cls_idx].long())
            else:
                loss = loss + nn.CrossEntropyLoss()(output[:, start_idx:end_idx], y.flatten().long())

        self.log('train_loss',  loss, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outs):
        epoch_acc = self.train_acc.compute()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)

        loss = 0
        class_dims = self.class_dims
        for cls_idx in range(len(class_dims)):
            start_idx =  sum(class_dims[:cls_idx])
            end_idx =  start_idx + class_dims[cls_idx]

            if y.ndim > 1:
                loss = loss + nn.CrossEntropyLoss()(output[:, start_idx:end_idx], y[:, cls_idx].long())
            else:
                loss = loss + nn.CrossEntropyLoss()(output[:, start_idx:end_idx], y.flatten().long())

        self.log('valid_loss', loss, prog_bar=True)
        #self.log('valid_acc', self.valid_acc(torch.argmax(output, dim=1), y), prog_bar=True)
        
        return loss

    def validation_epoch_end(self, outs):
        self.valid_acc.compute()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer