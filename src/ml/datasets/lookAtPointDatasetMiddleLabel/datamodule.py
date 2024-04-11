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
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from lightning.pytorch import LightningDataModule

# from dataset import LookAtPointDataset
from ml.datasets.lookAtPointDatasetMiddleLabel.dataset import (
    LookAtPointDatasetMiddleLabel,
)


class LookAtPointDataMiddleLabelModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/home/martin/Documents/Exjobb/eed/.data/",
        sklearn: bool = False,
        batch_size: int = 32,
        validation_split: float = 0.2,
        num_workers: int = 0,
        window_size: int = 250,
        print_extractionTime: bool = False,

    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_workers = num_workers
        self.sklearn = sklearn
        self.window_size = window_size
        self.print_extractionTime = print_extractionTime

    def setup(self, stage=None):
        # Load dataset
        dataset = LookAtPointDatasetMiddleLabel(self.data_dir, self.window_size, self.print_extractionTime)
        self.dataset = dataset
        print("sklearn:", self.sklearn)
        data_len = len(dataset)
        print(data_len)
        if self.sklearn:
            print("Using sklearn")
            self.batch_size = data_len

        # Calculate sizes of train/validation split
        val_size = int(data_len * self.validation_split)
        train_size = data_len - val_size

        # Split dataset
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
