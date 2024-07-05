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
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms

# from dataset import LookAtPointDataset
from ml.datasets.lookAtPointDatasetMiddleLabel.dataset import (
    LookAtPointDatasetMiddleLabel,
)


def custom_collate_fn(batch):
    pre_window_data = [item[0] for item in batch]
    center_window_data = [item[1] for item in batch]
    post_window_data = [item[2] for item in batch]
    target = [item[3] for item in batch]

    pre_data = np.array(pre_window_data)
    center_data = np.array(center_window_data)
    post_data = np.array(post_window_data)

    features = extract_features_batchwise(pre_data, center_data, post_data)

    return [features, target]


def extract_features_batchwise(pre_data, center_data, post_data):
    features = {}
    features["mean-diff"] = np.hypot(
        np.mean(post_data[:, 0], axis=1) - np.mean(pre_data[:, 0], axis=1),
        np.mean(post_data[:, 1], axis=1) - np.mean(pre_data[:, 1], axis=1),
    )
    features["med-diff"] = np.hypot(
        np.median(post_data[:, 0], axis=1) - np.median(pre_data[:, 0], axis=1),
        np.median(post_data[:, 1], axis=1) - np.median(pre_data[:, 1], axis=1),
    )

    features["std-diff"] = np.hypot(
        np.std(post_data[:, 0], axis=1), np.std(post_data[:, 1], axis=1)
    ) - np.hypot(np.std(pre_data[:, 0], axis=1), np.std(pre_data[:, 1], axis=1))

    features["std"] = np.hypot(
        np.std(center_data[:, 0], axis=1), np.std(center_data[:, 1], axis=1)
    )

    return features


class LookAtPointDataMiddleLabelModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/home/martin/Documents/Exjobb/eed/.data/",
        sklearn: bool = False,
        batch_size: int = 32,
        validation_split: float = 0.2,
        num_workers: int = 15,
        window_size: int = 100,
        window_size_vel: int = 20,
        window_size_dir: int = 22,
        print_extractionTime: bool = False,
        max_presaved_epochs: int = 99,
        noise_levels: list = [0],
        training_datasets: list[str] = [],#["lund2013"],
        savgol_filter_window :int =10, 
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_workers = num_workers
        self.sklearn = sklearn
        self.window_size = window_size
        self.window_size_vel = window_size_vel
        self.window_size_dir = window_size_dir
        self.print_extractionTime = print_extractionTime
        self.max_presaved_epochs = max_presaved_epochs
        self.noise_levels = noise_levels
        self.training_datasets = training_datasets
        self.savgol_filter_window = savgol_filter_window 

    def setup(self, stage=None):
        # Load dataset
        self.train_dataset = LookAtPointDatasetMiddleLabel(
            data_dir=self.data_dir,
            long_window_size=self.window_size,
            window_size_vel=self.window_size_vel,
            window_size_dir=self.window_size_dir,
            print_extractionTime=self.print_extractionTime,
            max_presaved_epoch=self.max_presaved_epochs,
            trainer=self.trainer,
            noise_levels=self.noise_levels,
            split="train",
            sklearn=self.sklearn,
            training_datasets=self.training_datasets,
            savgol_filter_window = self.savgol_filter_window
        )
        print("sklearn:", self.sklearn)
        data_len = len(self.train_dataset)
        print("Train data length:", data_len)

        ### Not necessary anymore because we are doing subject-level splits
        # Calculate sizes of train/validation split
        # val_size = int(data_len * self.validation_split)
        # train_size = data_len - val_size

        # Split dataset
        # self.train_dataset, self.val_dataset = random_split(
        #    dataset, [train_size, val_size]
        # )

        self.val_dataset = LookAtPointDatasetMiddleLabel(
            self.data_dir,
            self.window_size,
            self.window_size_vel,
            self.window_size_dir,
            self.print_extractionTime,
            self.max_presaved_epochs,
            self.trainer,
            self.noise_levels,
            split="val",
            sklearn=self.sklearn,
            training_datasets=self.training_datasets,
            savgol_filter_window=self.savgol_filter_window
        )

        self.test_dataset = LookAtPointDatasetMiddleLabel(
            self.data_dir,
            self.window_size,
            self.window_size_vel,
            self.window_size_dir,
            self.print_extractionTime,
            self.max_presaved_epochs,
            self.trainer,
            self.noise_levels,
            split="test",
            sklearn=self.sklearn,
            training_datasets=self.training_datasets,
            savgol_filter_window=self.savgol_filter_window

        )

        if self.sklearn:  # make sure that we always fetch
            # all of the data when using sklearn
            print("Using sklearn")
            self.batch_size = max(
                len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)
            )

    def train_dataloader(self):
        print("number of workers:", self.num_workers)
        return DataLoader(
            self.train_dataset,  # collate_fn=custom_collate_fn,
            batch_size=self.batch_size,
            shuffle=not (self.sklearn),
            num_workers=0#self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=not (self.sklearn),
            num_workers=0#self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=not (self.sklearn),
            num_workers=0#self.num_workers,
        )
