"""
Replace the following code with your own LightningDataModule class.
Reference: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
Example:

from lightning.pytorch import LightningDataModule


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False)
        self.mnist_predict = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
import pytorch_lightning as pl
from lightning.pytorch import LightningDataModule

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)
print(project_root)
# from dataset import LookAtPointDataset
from ml.datasets.lookAtPointDataset.dataset import LookAtPointDataset


class LookAtPointDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/home/martin/Documents/Exjobb/eed/.data/",
        sklearn: bool = True,
        batch_size: int = 32,
        validation_split: float = 0.2,
        num_workers: int = 0,
        window_size: int = 250,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_workers = num_workers
        self.sklearn = sklearn
        self.window_size = window_size

    def setup(self, stage=None):
        # Load dataset
        dataset = LookAtPointDataset(self.data_dir, self.window_size, stride=1)
        self.dataset = dataset
        data_len = len(dataset)

        if self.sklearn:
            self.batch_size = data_len

        # Calculate sizes of train/validation split
        val_size = int(data_len * self.validation_split)
        train_size = data_len - val_size

        # Split dataset
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


# #Code to check that the data has been loaded correctly
# data_dir = '/home/martin/Documents/Exjobb/eed/.data'
# batch_size = 32
# validation_split = 0.2
# num_workers = 4
#
# data_module = LookAtPointDataModule(data_dir, batch_size=batch_size, validation_split=validation_split, num_workers=num_workers)
## Call setup method
# data_module.setup()
#
## Check dataset sizes
# print(f"Training dataset size: {len(data_module.train_dataset)}")
# print(f"Validation dataset size: {len(data_module.val_dataset)}")
#
## Verify DataLoader creation
# train_dataloader = data_module.train_dataloader()
# val_dataloader = data_module.val_dataloader()
#
## Check sample batches
# for batch_idx, (x_batch, y_batch, evt_batch) in enumerate(train_dataloader):
#   print(f"Train Batch {batch_idx}:")
#   print(f"  x: {x_batch}")
#   print(f"  y: {y_batch}")
#   print(f"  evt: {evt_batch}")
#   # Print a few batches for brevity
#   if batch_idx >= 2:
#       break
#
# for batch_idx, (x_batch, y_batch, evt_batch) in enumerate(val_dataloader):
#   print(f"Validation Batch {batch_idx}:")
#   print(f"  x: {x_batch}")
#   print(f"  y: {y_batch}")
#   print(f"  evt: {evt_batch}")
#   # Print a few batches for brevity
#   if batch_idx >= 2:
#       break
# print("hej")
