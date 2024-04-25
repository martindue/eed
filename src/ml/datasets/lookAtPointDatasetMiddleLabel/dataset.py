import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import os, time
import pandas as pd
import scipy.signal as sg
import astropy.stats as ast
from tqdm import tqdm


x = os.path.dirname(__file__)
from ...utils.helpers import vcorrcoef


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
        post_idx = [slice(cumulative_index + indices[s], cumulative_index + indices[s + window_size - 1] + 1) for s in onset]
        pre_idx = [slice(cumulative_index + indices[s - window_size], cumulative_index + indices[s - 1] + 1) for s in onset]
        center_idx = [
            slice(cumulative_index + indices[s - window_size // 2], cumulative_index + indices[s + window_size // 2] + 1)
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
        print_extractionTime: bool = False,  # TODO:remove
        max_presaved_epoch: int = 98,
        trainer: pl.Trainer = None,
        noise_levels: list = [0, 0.1],
        train: bool = True,
        sklearn: bool = False,
    ):
        self.noise_levels = noise_levels
        self.print_extractionTime = print_extractionTime
        self.data_dir = data_dir
        self.max_presaved_epoch = max_presaved_epoch
        self.trainer = trainer
        self.train = train
        self.data = None
        self.data_df = None
        self.labels = None
        self.pre_window_indices = None
        self.center_window_indices = None
        self.post_window_indices = None
        self.presaved_features = None
        self.file_names = []
        self.sklearn = sklearn

        self.load_data()
        self.interpolate_nan_values()
        self.setup_window_indices(long_window_size)
        self.setup_augmented_data()

    def load_data(self):
        if self.train:
            load_dir = os.path.join(self.data_dir, "raw/train_data")
        else:
            load_dir = os.path.join(self.data_dir, "raw/test_data")

        file_list = os.listdir(load_dir)
        numpy_files = [f for f in file_list if f.endswith(".npy")]
        self.file_names = file_list
        appended_df = None
        for idx, file_name in enumerate(numpy_files):
            file_path = os.path.join(load_dir, file_name)
            loaded_array = np.load(file_path)
            df = pd.DataFrame(loaded_array)
            df["file_index"] = idx
            if appended_df is None:
                appended_df = df
            else:
                appended_df = pd.concat((appended_df, df))

        # appended_df = df  # Remove this line to use the full dataset.
        self.data_df = appended_df
        self.data = np.stack((appended_df["x"], appended_df["y"]), axis=-1)
        self.labels = appended_df["evt"]

    def interpolate_nan_values(self):
        for i in range(self.data.shape[1]):
            nan_indices = np.isnan(self.data[:, i])
            if np.any(nan_indices):
                self.data[:, i][nan_indices] = np.interp(
                    np.flatnonzero(nan_indices),
                    np.flatnonzero(~nan_indices),
                    self.data[:, i][~nan_indices],
                )

    def setup_window_indices(self, long_window_size: int):
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
        aug_dir = os.path.join(self.data_dir, "augmented_data")

        if not os.path.exists(aug_dir):
            os.makedirs(aug_dir)

        noise_levels = [0, 0.1]
        for i, noise_level in enumerate(noise_levels):
            if not any(
                f"augmented_data_{i}_noise_{noise_level}" in file
                for file in os.listdir(aug_dir)
            ):
                print("Noise level ", noise_level)
                # add noise and extract features
                aug_data = add_normal_noise(self.data, noise_level)
                dict_list = []
                for idx in tqdm(
                    range(len(self.center_window_indices)), desc="Extracting features"
                ):
                    pre_window = self.pre_window_indices[idx]
                    center_window = self.center_window_indices[idx]
                    post_window = self.post_window_indices[idx]
                    pre_window_data = aug_data[pre_window]
                    center_window_data = aug_data[center_window]
                    post_window_data = aug_data[post_window]
                    features = self.extract_features(
                        center_window,
                        pre_window_data,
                        center_window_data,
                        post_window_data,
                    )

                    dict_list.append(features)
                features_df = pd.DataFrame.from_records(dict_list)
                # save augmentation as parquet
                features_df.to_parquet(
                    os.path.join(
                        aug_dir, f"augmented_data_{i}_noise_{noise_level}.parquet"
                    )
                )

        # load parquet files and append them to the original data
        df_list = None
        for file in os.listdir(aug_dir):
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
        features = self.extract_features(
            center_window,
            pre_window_data,
            center_window_data,
            post_window_data,
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
                "label": torch.tensor(
                    self.presaved_features.iat[idx, 4], dtype=torch.long
                ),
                "t": self.presaved_features.iat[idx, 0],
                "x": self.presaved_features.iat[idx, 1],
                "y": self.presaved_features.iat[idx, 2],
                "status": self.presaved_features.iat[idx, 3],
                "file_name": self.file_names[self.presaved_features.iat[idx, 5]],
                "file_index": self.presaved_features.iat[idx, 5],
            }

    def extract_features(
        self, center_window, pre_window_data, center_window_data, post_window_data
    ):
        pre_x_windowed = pre_window_data[:, 0]
        pre_y_windowed = pre_window_data[:, 1]

        c_x_windowed = center_window_data[:, 0]
        c_y_windowed = center_window_data[:, 1]
        c_x_windowed_dx = np.diff(c_x_windowed)
        c_y_windowed_dy = np.diff(c_y_windowed)

        post_x_windowed = post_window_data[:, 0]
        post_y_windowed = post_window_data[:, 1]

        middlePoint = (center_window.start + center_window.stop) // 2

        label = self.labels.values[
            middlePoint
        ]  # The label corresponds to the middle element in the center window

        t = self.data_df.iat[middlePoint, 0]
        x = self.data_df.iat[middlePoint, 1]
        y = self.data_df.iat[middlePoint, 2]
        status = self.data_df.iat[middlePoint, 3]
        file_index = self.data_df.iat[middlePoint, 5]

        features = dict()

        features["t"] = t  # 0
        features["x"] = x  # 1
        features["y"] = y  # 2
        features["status"] = status  # 3
        features["label"] = label  # 4
        features["file_index"] = file_index  # 5
        features["file_name"] = self.file_names[file_index]  # 6

        fs = 1000  # TODO: Extract fs from timestamps in data.
        features["fs"] = fs

        for d, dd in zip(
            ["x", "y"],
            [(pre_x_windowed, post_x_windowed), (pre_y_windowed, post_y_windowed)],
        ):
            # difference between positions of preceding and succeding windows,
            # aka tobii feature, together with data quality features and its variants
            features["mean-diff-%s" % d] = np.nanmean(dd[0]) - np.nanmean(dd[1])
            features["med-diff-%s" % d] = np.nanmedian(dd[0]) - np.nanmedian(dd[1])

            # standard deviation
            features["std-pre-%s" % d] = np.nanstd(dd[0])
            features["std-post-%s" % d] = np.nanstd(dd[1])

        features["std-x"] = np.nanstd(c_x_windowed)
        features["std-y"] = np.nanstd(c_y_windowed)

        features["std"] = np.hypot(features["std-x"], features["std-y"])
        features["std-diff"] = np.hypot(
            features["std-post-x"], features["std-post-y"]
        ) - np.hypot(features["std-pre-x"], features["std-pre-y"])

        # bcea
        bcea_pre = calculate_bcea(pre_x_windowed, pre_y_windowed)
        bcea_center = calculate_bcea(c_x_windowed, c_y_windowed)
        bcea_post = calculate_bcea(post_x_windowed, post_y_windowed)

        features["bcea"] = bcea_center
        features["bcea_diff"] = bcea_post - bcea_pre

        # RMS
        features["rms"] = np.hypot(
            np.sqrt(np.mean(np.square(c_x_windowed))),
            np.sqrt(np.mean(np.square(c_y_windowed))),
        )
        features["rms-diff"] = np.hypot(
            np.sqrt(np.mean(np.square(post_x_windowed))),
            np.sqrt(np.mean(np.square(post_y_windowed))),
        ) - np.hypot(
            np.sqrt(np.mean(np.square(pre_x_windowed))),
            np.sqrt(np.mean(np.square(pre_y_windowed))),
        )

        # dispersion, aka idt feature
        x_range = np.nanmax(c_x_windowed) - np.nanmin(c_x_windowed)
        y_range = np.nanmax(c_y_windowed) - np.nanmin(c_y_windowed)
        features["disp"] = x_range + y_range

        # velocity and acceleration #TODO: fix this. Make window length indep of normal window length. These two features take half the time to compute as the rest of the features.
        # features["vel"] = np.hypot(sg.savgol_filter(c_x_windowed_dx, len(c_x_windowed_dx)-1, 2,1), sg.savgol_filter(c_y_windowed_dy, len(c_y_windowed)-1, 2,1))*fs
        # features["acc"] = np.hypot(sg.savgol_filter(c_x_windowed_dx, len(c_x_windowed_dx)-1, 2,2), sg.savgol_filter(c_y_windowed_dy, len(c_y_windowed)-1, 2,2))*fs**2
        features["vel"] = np.mean(
            np.hypot(np.diff(c_x_windowed), np.diff(c_y_windowed)) * fs
        )
        features["acc"] = np.mean(
            np.hypot(np.diff(np.diff(c_x_windowed)), np.diff(np.diff(c_y_windowed)))
            * fs**2
        )
        # rayleightest
        angl = np.arctan2(c_y_windowed, c_x_windowed)
        features["rayleightest"] = ast.rayleightest(angl)
        return features
