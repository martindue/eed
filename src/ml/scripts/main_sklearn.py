# https://git.smarteye.se/research/DepthSkeleton/-/blob/main/training/config/showData.yaml?ref_type=heads

# https://git.smarteye.se/research/DepthSkeleton/-/blob/main/training/showData.py?ref_type=heads
import os, sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)

from ml.datasets.lookAtPointDatasetNoWindow.datamodule import (
    LookAtPointDataModuleNoWindow,
)
from ml.datasets.lookAtPointDataset.datamodule import LookAtPointDataModule
from ml.datasets.lookAtPointDatasetMiddleLabel.datamodule import (
    LookAtPointDataMiddleLabelModule,
)
from jsonargparse import CLI, ArgumentParser
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from lightning.pytorch import LightningDataModule
import numpy as np
from typing import Type


def main(
    data_module: LightningDataModule = LookAtPointDataModule(),
    stage: str = "fit",
    classifier_type: Type[BaseEstimator] = RandomForestClassifier,
    rf_config: dict = {
        "n_trees": 100,
        "max_depth": None,
        "class_weight": "balanced_subsample",
        "max_features": 3,
        "n_jobs": 1,
        "verbose": 3,
    },
):
    data_module.prepare_data()
    data_module.setup(stage=stage)
    data_loader = data_module.train_dataloader()

    for data in data_loader:
        X = data["input"]
        X = X.squeeze()
        y = data["evt"]

    if classifier_type == RandomForestClassifier:
        clf = RandomForestClassifier(
            n_estimators=rf_config["n_trees"],
            max_depth=rf_config["max_depth"],
            class_weight=rf_config["class_weight"],
            max_features=rf_config["max_features"],
            n_jobs=rf_config["n_jobs"],
            verbose=rf_config["verbose"],
        )
    X = X.numpy()
    y = y.numpy()
    X = np.nan_to_num(
        X, nan=0
    )  # placeholder for missing values TODO: change to something more sensible
    print("dimensions of X and y:", X.shape, y.shape)
    clf.fit(X, y)
    # print(data)


if __name__ == "__main__":
    CLI(main)
