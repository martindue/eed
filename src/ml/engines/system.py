"""
Replace the following code with your own LightningModule class.
Reference: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
Example:

import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F


class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class LitModel(pl.LightningModule):
    def __init__(self, model):
        """
        Initializes the LightningModule with the provided model.

        Args:
            model: The PyTorch model to be trained.
        """
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
            batch: A batch of data from the dataloader.
            batch_idx: The index of the batch.

        Returns:
            The loss tensor for the current batch.
        """
        x = batch['input']
        evt = batch['evt']
        evt_hat = self.model(x)
        loss = F.cross_entropy(evt_hat, evt)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer(s) for training.

        Returns:
            The optimizer(s) to be used for training.
        """
        optimizer = torch.optim.Adam(self.model.parameters())
        return optimizer
