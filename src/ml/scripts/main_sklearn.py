# https://git.smarteye.se/research/DepthSkeleton/-/blob/main/training/config/showData.yaml?ref_type=heads

# https://git.smarteye.se/research/DepthSkeleton/-/blob/main/training/showData.py?ref_type=heads
import os, sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)


from ml.datasets.lookAtPointDatasetMiddleLabel.datamodule import (
    LookAtPointDataMiddleLabelModule,
)
from jsonargparse import CLI, ArgumentParser
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from lightning.pytorch import LightningDataModule
import numpy as np
import pandas as pd
from itertools import chain
from collections import OrderedDict
from typing import Type
from ml.utils.helpers import impute_with_column_means

import numpy as np


# Example usage:
# X_imputed = impute_with_column_means(X)

def fetch_data(data_loader):
    t = np.concatenate([batch["t"].numpy() for batch in data_loader], axis=0)
    xx = np.concatenate([batch["x"].numpy() for batch in data_loader], axis=0)
    yy = np.concatenate([batch["y"].numpy() for batch in data_loader], axis=0)
    status = np.concatenate([batch["status"].numpy() for batch in data_loader], axis=0)
    gt_label = np.concatenate([batch["label"].numpy() for batch in data_loader], axis=0)
    file_index = np.concatenate(
        [batch["file_index"].numpy() for batch in data_loader], axis=0
    )
    pd_output_df = pd.DataFrame(
        {
            "t": t,
            "x": xx,
            "y": yy,
            "status": status,
            "evt": gt_label,
            "file_index": file_index,
        }
    )

    return pd_output_df

def make_predictions_and_save(classifier, X,output_df, output_dir):
    # Make predictions
    y_pred = classifier.predict(X)

    #get unique file names from output_df
    unique_file_names = output_df["file_name"].unique()


    file_index = output_df["file_index"].values
    gt_output_df = output_df.drop(columns=["file_index","file_name"])
    unique_file_indices = output_df["file_index"].unique()
    print("unique_file_indices: ", unique_file_indices)

    pd_output_df = output_df.drop(columns=["file_index", "evt", "file_name"])
    pd_output_df["evt"] = y_pred

    # Save predictions for each unique file index
    for index in unique_file_indices:
        # Filter samples belonging to the current file index
        mask = file_index == index
        pd_filtered_df = pd_output_df[mask]
        gt_filtered_df = gt_output_df[mask]
        file_name = os.path.splitext(unique_file_names[index])[0]
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save predictions to a file
        print(f"Saving predictions for {file_name} to {output_dir}...")
        pd_file_path = os.path.join(output_dir, f"{file_name}_pd.csv")
        pd_filtered_df.sort_values(by=["t"], inplace=True)
        pd_filtered_df.to_csv(pd_file_path, index=False)

        gt_file_path = os.path.join(output_dir, f"{file_name}_gt.csv")
        gt_filtered_df.sort_values(by=["t"], inplace=True)
        gt_filtered_df.to_csv(gt_file_path, index=False)


def main(
    data_module: LightningDataModule = LookAtPointDataMiddleLabelModule(),
    stage: str = "fit",
    classifier_type: Type[BaseEstimator] = RandomForestClassifier,
    rf_config: dict = {
        "n_trees": 100,
        "max_depth": None,
        "class_weight": "balanced_subsample",
        "max_features": 3,
        "n_jobs": 10,
        "verbose": 3,
    },
):
    data_module.prepare_data()
    data_module.setup(stage=stage)
    train_data_loader = data_module.train_dataloader()
    validation_data_loader = data_module.val_dataloader()

    if classifier_type == RandomForestClassifier:
        clf = RandomForestClassifier(
            n_estimators=rf_config["n_trees"],
            max_depth=rf_config["max_depth"],
            class_weight=rf_config["class_weight"],
            max_features=rf_config["max_features"],
            n_jobs=rf_config["n_jobs"],
            verbose=rf_config["verbose"],
        )

    for batch in train_data_loader:
        X_train = batch["features"]
        y_train = batch["label"]
        t =  batch["t"]
        xx = batch["x"]
        yy = batch["y"]
        status = batch["status"]
        file_index = batch["file_index"]
        file_name = batch["file_name"]
    train_output_df = pd.DataFrame(
        {
            "t": t,
            "x": xx,
            "y": yy,
            "status": status,
            "evt": y_train,
            "file_index": file_index,
            "file_name": file_name,
        }
    )
    
    for batch in validation_data_loader:
        X_val = batch["features"]
        t = batch["t"]
        xx = batch["x"]
        yy = batch["y"]
        y_val = batch["label"]
        status = batch["status"]
        file_index = batch["file_index"]
        file_name = batch["file_name"]

    val_output_df = pd.DataFrame(
        {
            "t": t,
            "x": xx,
            "y": yy,
            "status": status,
            "evt": y_val,
            "file_index": file_index,
            "file_name": file_name, 
        }
    )
    
    X_train = impute_with_column_means(X_train)
    X_val = impute_with_column_means(X_val)

    print("dimensions of X and y:", X_train.shape, y_train.shape)
    print("Fitting classifier.... ")
    clf.fit(X_train, y_train)


    # Make predictions and save them
    print("Predicting on train data....")
    make_predictions_and_save(
        clf, X_train, train_output_df, ".experiments/results/sklearn/train"
    )
    print("Predicting on validation data....")
    make_predictions_and_save(
        clf, X_val, val_output_df, ".experiments/results/sklearn/validation"
    )


if __name__ == "__main__":
    CLI(main)
