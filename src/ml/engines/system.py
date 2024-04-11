"""
Replace the following code with your own LightningModule class.
Reference: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
Example:

import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F


class LitModule(pl.LightningModule):
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
# import pytorch_lightning
from lightning.pytorch import LightningModule
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LitModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: pl.cli.OptimizerCallable = torch.optim.Adam,
        scheduler: pl.cli.LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
    ) -> None:
        """
        Initializes the LightningModule with the provided model.

        Args:
            model: The PyTorch model to be trained.
        """
        super().__init__()
        self.model = model
        print("model: ", model)
        print("optimizer: ", optimizer)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
            batch: A batch of data from the dataloader.
            batch_idx: The index of the batch.

        Returns:
            The loss tensor for the current batch.
        """

        print("Current device:", self.device)
        features, y = batch

        input = torch.stack([features[feature].float() for feature in features], dim=1)
        y = y.float()

        y_hat = self(input)

        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Configures the optimizer(s) for training.

        Returns:
            The optimizer(s) to be used for training.
        """
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
