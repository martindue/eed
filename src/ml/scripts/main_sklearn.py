# https://git.smarteye.se/research/DepthSkeleton/-/blob/main/training/config/showData.yaml?ref_type=heads

# https://git.smarteye.se/research/DepthSkeleton/-/blob/main/training/showData.py?ref_type=heads
import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)


from collections import OrderedDict
from itertools import chain
from typing import Type

import numpy as np
import pandas as pd
from jsonargparse import CLI, ArgumentParser
from lightning.pytorch import LightningDataModule
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from ml.datasets.lookAtPointDatasetMiddleLabel.datamodule import (
    LookAtPointDataMiddleLabelModule,
)
from ml.utils.helpers import impute_with_column_means, linear_interpol_with_pandas
from ml.utils.post_processing import post_process, fixation_merge


# Example usage:
# X_imputed = impute_with_column_means(X)


def make_predictions_and_save(classifier, X, output_df, output_dir, pp_args):
    # Make predictions
    y_pred = classifier.predict(X)

    # get unique file names from output_df
    unique_file_names = output_df["file_name"].unique()

    file_index = output_df["file_index"].values
    gt_output_df = output_df.drop(columns=["file_index", "file_name"])
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

        print(f"Saving predictions for {file_name} to {output_dir}...")
        pd_file_path = os.path.join(output_dir, f"{file_name}_pd.csv")
        #        pd_filtered_df = pd_filtered_df.sort_values(by=["t"]) #should be unnecessary

        # post process the predictions
        pd_pp_df = post_process(pd_filtered_df, pp_args)
        pd_pp_df = pd_pp_df.set_index(pd_filtered_df.index)

        # Temporary removal of samples made false by post processing
        mask2 = pd_pp_df["status"] == True
        pd_pp_df = pd_pp_df[mask2]
        gt_filtered_df = gt_filtered_df[mask2]

        # Save predictions to a file
        pd_pp_df.to_csv(pd_file_path, index=False)

        gt_file_path = os.path.join(output_dir, f"{file_name}_gt.csv")
        #        gt_filtered_df = gt_filtered_df.sort_values(by=["t"]) #should be unnecessary
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
    pp_args: dict = {
        "sacc_minimum_distance": 0.1,
        "pp_kwargs": {
            "thres_id": 75.0,
            "thres_ifa": 0.2,
            "thres_ifi": 75.0,
            "thres_sd_s": 3,
            "thres_pd": 3,
            "thres_isi": 25.0,
            "thres_sd_lo": 6.0,
            "thres_sd_hi": 150.0,
            "thres_fd": 50.0,
        },
    },
    pre_proc_args: dict = { 
        "sacc_minimum_distance": 1,
    },
    event_map: dict= {
        "1": 1,
        "2": 2,
        "3": 1,
        "4": 1,
        "5": 5,
        "0": 0
    }
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
        t = batch["t"]
        xx = batch["x"]
        yy = batch["y"]
        status = batch["status"]
        file_index = batch["file_index"]
        file_name = batch["file_name"]
    train_df = pd.DataFrame(
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
    train_df.replace({"evt": event_map}, inplace=True)
    train_mask = ((train_df["status"]==True)&(train_df["evt"] != 5))&(train_df["evt"] != 0)
    train_df = train_df[train_mask]
    y_train = y_train[train_mask]
    X_train = X_train[train_mask] # this is slightly wrong, because the window can still contain samples with status False

    train_df.reset_index(drop=True, inplace=True)
    train_df = fixation_merge(train_df, pre_proc_args["sacc_minimum_distance"])
    for batch in validation_data_loader:
        X_val = batch["features"]
        t = batch["t"]
        xx = batch["x"]
        yy = batch["y"]
        y_val = batch["label"]
        status = batch["status"]
        file_index = batch["file_index"]
        file_name = batch["file_name"]

    val_df = pd.DataFrame(
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
    val_df.replace({"evt": event_map}, inplace=True)
    val_mask = ((val_df["status"]==True)&(val_df["evt"]!=5))&(val_df["evt"]!=0)
    val_df = val_df[val_mask]
    y_val = y_val[val_mask]
    X_val = X_val[val_mask] # this is slightly wrong, because the window can still contain samples with status False

    
    val_df.reset_index(drop=True, inplace=True)

    val_df = fixation_merge(val_df, pre_proc_args["sacc_minimum_distance"])


    # "sort the dataframes by file_index and t"  # This is dangerous, because X_train and X_val are not sorted
    # train_output_df = train_output_df.sort_values(by=["file_index", "t"])
    # val_output_df = val_output_df.sort_values(by=["file_index", "t"])

    # X_train = train_output_df.pop("X_train")
    # X_val = val_output_df.pop("X_val")

   #X_train = impute_with_column_means(X_train)
    #X_val = impute_with_column_means(X_val)
    X_train = linear_interpol_with_pandas(X_train)
    X_val = linear_interpol_with_pandas(X_val)

    print("dimensions of X and y:", X_train.shape, y_train.shape)
    print("Fitting classifier.... ")
    clf.fit(X_train, y_train)

    # save the classifier
    import joblib
    joblib.dump(clf, "classifier.joblib")


    # Make predictions and save them
    print("Predicting on train data....")
    make_predictions_and_save(
        clf,
        X_train,
        train_df,
        ".experiments/results/sklearn/train",
        pp_args=pp_args,
    )
    print("Predicting on validation data....")
    make_predictions_and_save(
        clf,
        X_val,
        val_df,
        ".experiments/results/sklearn/validation",
        pp_args=pp_args,
    )


if __name__ == "__main__":
    CLI(main)
