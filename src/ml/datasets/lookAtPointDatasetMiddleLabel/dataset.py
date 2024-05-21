import os
import time

import astropy.stats as ast
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.signal as sg
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

x = os.path.dirname(__file__)
print(x)
from ...utils.helpers import vcorrcoef_rolling


def get_window_indices(data, window_params):
    window_size = window_params["window_size"]
    window_stride = window_params["window_resolution"]

    post_idx_accum = []
    pre_idx_accum = []
    center_idx_accum = []
    cumulative_index = 0  # Initialize cumulative index

    for _, d in data.groupby("file_index"):
        # get indices for each time window
        indices = d.index.values

        onset = range(window_size, len(d) - window_size, window_stride)
        post_idx = [
            slice(
                cumulative_index + indices[s],
                cumulative_index + indices[s + window_size - 1] + 1,
            )
            for s in onset
        ]
        pre_idx = [
            slice(
                cumulative_index + indices[s - window_size],
                cumulative_index + indices[s - 1] + 1,
            )
            for s in onset
        ]
        center_idx = [
            slice(
                cumulative_index + indices[s - window_size // 2],
                cumulative_index + indices[s + window_size // 2] + 1,
            )
            for s in onset
        ]

        post_idx_accum.extend(post_idx)
        pre_idx_accum.extend(pre_idx)
        center_idx_accum.extend(center_idx)

        # Update cumulative index
        cumulative_index += len(d)

    return pre_idx_accum, center_idx_accum, post_idx_accum


def calculate_bcea(x_windowed, y_windowed):
    P = 0.68
    k = np.log(1 / (1 - P))
    rho = vcorrcoef(x_windowed, y_windowed)
    bcea = (
        2
        * k
        * np.pi
        * np.nanstd(x_windowed)
        * np.nanstd(y_windowed)
        * np.sqrt(1 - np.power(rho, 2))
    )
    return bcea


def add_normal_noise(data, noise_level=0.1):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def add_uniform_noise(data, noise_level=0.1):
    noise = np.random(-noise_level, noise_level, data.shape)
    return data + noise


class LookAtPointDatasetMiddleLabel(Dataset):
    def __init__(
        self,
        data_dir: str = "/home/martin/Documents/Exjobb/eed/.data",
        long_window_size: int = 250,
        window_size_vel: int = 20,
        window_size_dir: int = 22,
        print_extractionTime: bool = False,  # TODO:remove
        max_presaved_epoch: int = 98,
        trainer: pl.Trainer = None,
        noise_levels: list = [0, 0.1],
        split: str = "train",
        sklearn: bool = False,
        training_datasets: list[str] = [],
        debugMode: bool = False,
    ):
        self.noise_levels = noise_levels
        self.print_extractionTime = print_extractionTime
        self.data_dir = data_dir
        self.max_presaved_epoch = max_presaved_epoch
        self.trainer = trainer
        self.split = split
        self.data = None
        self.data_df = None
        self.labels = None
        self.pre_window_indices = None
        self.center_window_indices = None
        self.post_window_indices = None
        self.window_size = long_window_size
        self.window_size_dir = window_size_dir
        self.window_size_vel = window_size_vel
        self.presaved_features = None
        self.file_names = []
        self.sklearn = sklearn
        self.training_datasets = training_datasets
        self.debugMode = debugMode
        self.fs = 1000

        self.load_data()
        self.interpolate_nan_values()
        self.setup_window_indices(long_window_size, window_size_vel, window_size_dir)
        self.setup_augmented_data()

    def load_data(self):
        if self.debugMode and self.split == "train":
            load_dirs = [os.path.join(self.data_dir, "raw/debugDir")]
        elif self.split == "train":
            load_dirs = [os.path.join(self.data_dir, "raw/train_data")]
            if self.training_datasets is not None and "lund2013" in self.training_datasets:
                load_dirs.append(os.path.join(self.data_dir, "raw/lund_2013"))
                print("Using Lund 2013 dataset")
        elif self.split == "val":
            load_dirs = [os.path.join(self.data_dir, "raw/val_data")]
        elif self.split == "test":
            load_dirs = [os.path.join(self.data_dir, "raw/test_data")]
        else:
            raise ValueError("Invalid split")

        if not any(os.path.exists(dir) and os.listdir(dir) for dir in load_dirs):
            raise FileNotFoundError("No valid load directories found")

        appended_df = None
        file_names = []
        file_idx = 0
        for load_dir in load_dirs:
            if os.path.exists(load_dir) and os.listdir(load_dir):
                file_list = os.listdir(load_dir)
                numpy_files = [f for f in file_list if f.endswith(".npy")]
                file_names.extend(file_list)
            for file_name in numpy_files:
                file_path = os.path.join(load_dir, file_name)
                loaded_array = np.load(file_path)
                df = pd.DataFrame(loaded_array)
                df["file_index"] = file_idx
                file_idx += 1
                if appended_df is None:
                    appended_df = df
                else:
                    appended_df = pd.concat((appended_df, df))
        # appended_df = df  # Remove this line to use the full dataset.
        self.data_df = appended_df
        self.data = np.stack((appended_df["x"], appended_df["y"]), axis=-1)
        self.labels = appended_df["evt"]
        self.file_names = file_names

    def interpolate_nan_values(self):
        for i in range(self.data.shape[1]):
            nan_indices = np.isnan(self.data[:, i])
            if np.any(nan_indices):
                self.data[:, i][nan_indices] = np.interp(
                    np.flatnonzero(nan_indices),
                    np.flatnonzero(~nan_indices),
                    self.data[:, i][~nan_indices],
                )

    def setup_window_indices(self, long_window_size: int, window_size_vel: int, window_size_dir: int):
        (
            self.pre_window_indices,
            self.center_window_indices,
            self.post_window_indices,
        ) = get_window_indices(
            self.data_df, {"window_size": long_window_size, "window_resolution": 1}
        )


    def __len__(self):
        return len(self.center_window_indices)  # number of windows

    # it returns a pandas df with the features
    def setup_augmented_data(self) -> pd.DataFrame:
        aug_dir = os.path.join(self.data_dir, "augmented_data", self.split)

        if not os.path.exists(aug_dir):
            os.makedirs(aug_dir)

        for i, noise_level in enumerate(self.noise_levels):
            if not any(
                f"augmented_data_{i}_noise_{noise_level}" in file
                for file in os.listdir(aug_dir)
            ):
                print("Noise level ", noise_level)
                # add noise and extract features
                aug_data = add_normal_noise(self.data, noise_level)
                feature_dict = {}
                dict_list = []

                features_df = self.extract_features(aug_data[:,0], aug_data[:,1])
                # save augmentation as parquet
                features_df.to_parquet(
                    os.path.join(
                        aug_dir,
                        f"augmented_data_{i}_noise_{noise_level}_{self.split}.parquet",
                    )
                )

        # load parquet files and append them to the original data
        df_list = None
        for file in os.listdir(aug_dir):
            if any(str(noise_level) in file for noise_level in self.noise_levels):
                file_path = os.path.join(aug_dir, file)
                df = pd.read_parquet(file_path)
                if df_list is None:
                    df_list = df
                else:
                    df_list = pd.concat((df_list, df))

        self.presaved_features = df_list
        return

    def online_augmentation(self, idx, noise_level=0.1):
        print("Online augmentation")
        data = add_normal_noise(self.data, noise_level)
        pre_window = self.pre_window_indices[idx]
        center_window = self.center_window_indices[idx]
        post_window = self.post_window_indices[idx]

        pre_window_data = data[pre_window]
        center_window_data = data[center_window]
        post_window_data = data[post_window]
        
        midpoint = (center_window.stop + center_window.start)//2
        dir_window_data = center_window_data[(midpoint-self.window_size_dir//2): (midpoint + self.window_size_dir//2)]
        vel_window_data = center_window_data[(midpoint-self.window_size_vel//2): (midpoint + self.window_size_vel//2)]

        features = self.extract_features(
            center_window,
            pre_window_data,
            center_window_data,
            post_window_data,
            dir_window_data,
            vel_window_data,
        )
        features_df = pd.DataFrame([features])
        return features_df

    def __getitem__(self, idx):
        # for later epochs, perform online augmentation
        if not self.sklearn and self.trainer.current_epoch >= self.max_presaved_epoch:
            features_df = self.online_augmentation(idx)
            return {
                "features": torch.squeeze(
                    torch.tensor(
                        features_df.loc[:, features_df.columns != "label"].values,
                        dtype=torch.float32,
                    )
                ),
                "label": torch.squeeze(
                    torch.tensor(features_df.loc[:, "label"], dtype=torch.long)
                ),
                "t": self.data_df.loc[idx, "t"],
                "x": self.data_df.loc[idx, "x"],
                "y": self.data_df.loc[idx, "y"],
                "status": self.data_df.loc[idx, "status"],
                "file_index": self.data_df.loc[idx, "file_index"],
                "file_name": self.data_df.loc[idx, "file_name"],
            }
        else:
            return {
                "features": torch.tensor(
                    self.presaved_features.iloc[idx, 7:],
                    dtype=torch.float32,
                ),
                "label": torch.tensor(self.presaved_features.iat[idx, 4], dtype=torch.long),
                "t": self.presaved_features.iat[idx, 0],
                "x": self.presaved_features.iat[idx, 1],
                "y": self.presaved_features.iat[idx, 2],
                "status": self.presaved_features.iat[idx, 3],
                "file_name": self.file_names[self.presaved_features.iat[idx, 5]],
                "file_index": self.presaved_features.iat[idx, 5],
            }
        
    def extract_features(self, x,y):
        x_df = pd.DataFrame(x)
        y_df = pd.DataFrame(y)
        feature_dict = {}
        # differentiate the data to get velocity and acceleration
        vel_data = np.hypot(
            sg.savgol_filter(x, self.window_size_vel, 2, 1, axis=0),
            sg.savgol_filter(y, self.window_size_vel, 2, 1, axis=0)
        )*self.fs
        acc_data = np.hypot(
            sg.savgol_filter(x, self.window_size_vel, 2, 2, axis=0),
            sg.savgol_filter(y, self.window_size_vel, 2, 2, axis=0)
        )*self.fs**2
        # take a moving average of the acceleration, using pandas
        acc_data_averaged = pd.DataFrame(acc_data).rolling(window=self.window_size_vel, center=True).mean().bfill().ffill().values
        
        feature_dict["t"] = self.data_df["t"].values
        feature_dict["x"] = self.data_df["x"].values
        feature_dict["y"] = self.data_df["y"].values
        feature_dict["status"] = self.data_df["status"].values
        feature_dict["label"] = self.labels.values
        feature_dict["file_index"] = self.data_df["file_index"].values
        feature_dict["file_name"] = [self.file_names[i] for i in self.data_df["file_index"].values]

        feature_dict["vel"] = vel_data
        feature_dict["acc"] = acc_data
        feature_dict["acc_averaged"] = acc_data_averaged.flatten()

        std_x = x_df.rolling(window=self.window_size, center=True).std().bfill().ffill().values.flatten()
        std_y = y_df.rolling(window=self.window_size, center=True).std().bfill().ffill().values.flatten()
        std = np.hypot(std_x, std_y)
        feature_dict["std"] = std
        feature_dict["std-diff"] = np.abs(np.roll(std, -(self.window_size-1)//2) - np.roll(std, (self.window_size-1)//2))

        mean_diff_x = x_df.shift(-self.window_size//2).rolling(window=self.window_size, center=True).mean() - x_df.shift(self.window_size//2).rolling(window=self.window_size, center=False).mean()
        mean_diff_y = y_df.shift(-self.window_size//2).rolling(window=self.window_size, center=True).mean() - y_df.shift(self.window_size//2).rolling(window=self.window_size, center=False).mean() 
        mean_diff = np.hypot(mean_diff_x.ffill().bfill().values.flatten(), mean_diff_y.ffill().bfill().values.flatten())

        feature_dict["mean-diff"] = mean_diff  

        med_diff_x = x_df.shift(-self.window_size//2).rolling(window=self.window_size, center=True).median() - x_df.shift(self.window_size//2).rolling(window=self.window_size, center=True).median()
        med_diff_y = y_df.shift(-self.window_size//2).rolling(window=self.window_size, center=True).median() - y_df.shift(self.window_size//2).rolling(window=self.window_size, center=True).median()
        med_diff = np.hypot(med_diff_x.ffill().bfill().values.flatten(), med_diff_y.ffill().bfill().values.flatten())

        feature_dict["med-diff"] = med_diff



        # bcea
        P = 0.68
        k = np.log(1 / (1 - P))
        rho = vcorrcoef_rolling(x_df, y_df,self.window_size)
        feature_dict["bcea"] = 2*k*np.pi*std_x*std_y* np.sqrt(1-np.power(rho,2)).values.flatten()
        
        #bcea_diff
        feature_dict['bcea-diff-directional'] = np.roll(feature_dict['bcea'], -(self.window_size-1)//2) - \
                    np.roll(feature_dict['bcea'], (self.window_size-1)//2)
        feature_dict['bcea-diff-abs'] = np.abs(feature_dict['bcea-diff-directional'])

        # RMS
        rms_x = np.sqrt(np.square(x_df).rolling(window=self.window_size, center=True).mean().bfill().ffill().values.flatten())
        rms_y = np.sqrt(np.square(y_df).rolling(window=self.window_size, center=True).mean().bfill().ffill().values.flatten())
        feature_dict["rms"] = np.hypot(rms_x, rms_y)

        feature_dict["rms-diff"] = np.roll(feature_dict["rms"], -(self.window_size-1)//2) - np.roll(feature_dict["rms"], (self.window_size-1)//2)

        # dispersion
        x_range = x_df.rolling(window=self.window_size, center=True).max().bfill().ffill().values.flatten() - x_df.rolling(window=self.window_size, center=True).min().bfill().ffill().values.flatten()
        y_range = y_df.rolling(window=self.window_size, center=True).max().bfill().ffill().values.flatten() - y_df.rolling(window=self.window_size, center=True).min().bfill().ffill().values.flatten()
        feature_dict["disp"] = x_range + y_range

        # rayleightest
        angl = np.arctan2(y_df.rolling(window=self.window_size_dir, center=True).mean().bfill().ffill().values.flatten(), x_df.rolling(window=self.window_size_dir, center=True).mean().bfill().ffill().values.flatten())
        feature_dict["rayleightest"] = ast.rayleightest(angl)

        features_df = pd.DataFrame.from_dict(feature_dict)

        return features_df

 