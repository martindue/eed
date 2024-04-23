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
import pandas as pd


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
        print("Current device: ", torch.cuda.get_device_name(0))
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.testing_predictions = []
        self.output_data = pd.DataFrame(
            columns=["t", "x", "y", "status", "evt", "ground_truth"]
        )

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
            batch: A batch of data from the dataloader.
            batch_idx: The index of the batch.

        Returns:
            The loss tensor for the current batch.
        """
        loss = self._shared_eval_step(batch, batch_idx)
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

    def _shared_eval_step(self, batch, batch_idx):
        data = batch
        input = data["features"]
        y = data["label"]

        y_hat = self(input)

        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Args:
            batch: A batch of data from the dataloader.
            batch_idx: The index of the batch.

        Returns:
            The loss tensor for the current batch.
        """
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Defines the test step.

        Args:
            batch: A batch of data from the dataloader.
            batch_idx: The index of the batch.

        Returns:
            The loss tensor for the current batch.
        """
        data = batch
        features = data["features"]
        y = data["label"]
        t = data["t"]
        xx = data["x"]
        yy = data["y"]
        status = data["status"]

        input = features

        y_hat = self(input)

        evt = torch.argmax(y_hat, dim=1)

        new_data = pd.DataFrame(
            {
                "t": t.cpu(),
                "x": xx.cpu(),
                "y": yy.cpu(),
                "status": status.cpu(),
                "evt": evt.cpu(),
                "ground_truth": y.cpu(),
            }
        )

        self.output_data = pd.concat([self.output_data, new_data])
        self.testing_predictions.append(y_hat)

        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        """
        Defines the predict step.

        Args:
            batch: A batch of data from the dataloader.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            The predictions for the current batch.
        """
        data = batch
        features = data["features"]  # TODO: remove t,x,y,status

        input = features

        y_hat = self(input)

        return y_hat

    def teardown(self, stage):
        if stage == "test":
            # self.testing_predictions = torch.cat(self.testing_predictions, dim=0)
            preds = torch.cat(self.testing_predictions, dim=0)
            number_encoded_preds = torch.argmax(preds, dim=1)
            # print("testing_predictions: ", number_encoded_preds.tolist())
            print(self.output_data)
            self.output_data.to_csv(".experiments/results/output_data.csv", index=False)
        return super().teardown(stage)
