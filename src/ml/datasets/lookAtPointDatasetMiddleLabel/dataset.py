import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import os, time
import pandas as pd
import scipy.signal as sg
import astropy.stats as ast


x = os.path.dirname(__file__)
from ...utils.helpers import vcorrcoef


def get_window_indices(data, window_params):
    window_size = window_params["window_size"]
    window_stride = window_params["window_resolution"]

    post_idx_accum = []
    pre_idx_accum = []
    center_idx_accum = []
    for _, d in data.groupby("file_index"):
        # get indices for each time window
        indices = d.index.values

        onset = range(window_size, len(d) - window_size, window_stride)
        post_idx = [slice(indices[s], indices[s + window_size - 1]) for s in onset]
        pre_idx = [slice(indices[s - window_size], indices[s - 1]) for s in onset]
        center_idx = [slice(indices[s - window_size // 2], indices[s + window_size // 2]) for s in onset]

        post_idx_accum.extend(post_idx)
        pre_idx_accum.extend(pre_idx)
        center_idx_accum.extend(center_idx)

    return pre_idx_accum, center_idx_accum, post_idx_accum


def calculate_bcea(x_windowed, y_windowed):
    P = 0.68
    k = np.log(1 / (1 - P))
    rho = vcorrcoef(x_windowed, y_windowed)
    bcea = 2 * k * np.pi * np.nanstd(x_windowed) * np.nanstd(y_windowed) * np.sqrt(1 - np.power(rho, 2))
    return bcea


class LookAtPointDatasetMiddleLabel(Dataset):
    def __init__(
        self,
        data_dir: str = "/home/martin/Documents/Exjobb/eed/.data",
        long_window_size: int = 250,
        short_window_size: int = 50,
        print_extractionTime: bool = False,

    ):
        self.print_extractionTime = print_extractionTime
        data_dir = data_dir + "/raw"
        file_list = os.listdir(data_dir)
        print("Files in data directory:", file_list)
        numpy_files = [f for f in file_list if f.endswith(".npy")]
        appended_df = None
        for idx, file_name in enumerate(numpy_files):
            file_path = os.path.join(data_dir, file_name)
            loaded_array = np.load(file_path)
            df = pd.DataFrame(loaded_array)
            df["file_index"] = idx
            if appended_df is None:
                appended_df = df
            else:
                appended_df = pd.concat((appended_df, df))

        # appended_df = loaded_array  # Remove this line to use the full dataset.
        print("Shape of data:", appended_df.shape)
        print("Columns in data:", appended_df.columns)
        print("head of appended_df: ", appended_df.head())
        t = appended_df["t"]
        x_data = appended_df["x"]
        y_data = appended_df["y"]
        status = appended_df["status"]
        evt = appended_df["evt"]
        labels = evt

        data = np.stack((x_data, y_data), axis=-1)

        # Interpolate NaN values for each column separately. Placeholder.
        for i in range(data.shape[1]):
            nan_indices = np.isnan(data[:, i])
            if np.any(nan_indices):
                data[:, i][nan_indices] = np.interp(
                    np.flatnonzero(nan_indices),
                    np.flatnonzero(~nan_indices),
                    data[:, i][~nan_indices],
                )

        self.data = data
        self.labels = labels
        (
            self.pre_window_indices,
            self.center_window_indices,
            self.post_window_indices,
        ) = get_window_indices(appended_df, {"window_size": long_window_size, "window_resolution": 1})
        # self.short_window_indices = get_window_indices(
        #    appended_df, {"window_size": short_window_size, "window_resolution": 1}
        # )
        # self.long_window_size = long_window_size

    def __len__(self):
        return len(self.center_window_indices)

    def __getitem__(self, idx):
        tic = time.time()
        pre_window = self.pre_window_indices[idx]
        center_window = self.center_window_indices[idx]
        post_window = self.post_window_indices[idx]

        pre_window_data = self.data[pre_window]
        center_window_data = self.data[center_window]
        post_window_data = self.data[post_window]

        label, features = self.extract_features(center_window, pre_window_data, center_window_data, post_window_data)

        toc = time.time()
        if self.print_extractionTime:
            print("Feature extraction took %.3f s." % (toc - tic))
        return features, torch.tensor(label.values)

    def extract_features(self, center_window, pre_window_data, center_window_data, post_window_data):
        pre_x_windowed = pre_window_data[:, 0]
        pre_y_windowed = pre_window_data[:, 1]

        c_x_windowed = center_window_data[:, 0]
        c_y_windowed = center_window_data[:, 1]
        c_x_windowed_dx = np.diff(c_x_windowed)
        c_y_windowed_dy = np.diff(c_y_windowed)

        post_x_windowed = post_window_data[:, 0]
        post_y_windowed = post_window_data[:, 1]

        middlePoint = (center_window.start + center_window.stop) // 2

        label = self.labels[middlePoint]
        # The label corresponds to the middle element in the center window

        features = dict()

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
        features["std-diff"] = np.hypot(features["std-post-x"], features["std-post-y"]) - np.hypot(
            features["std-pre-x"], features["std-pre-y"]
        )

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
        #features["vel"] = np.hypot(sg.savgol_filter(c_x_windowed_dx, len(c_x_windowed_dx)-1, 2,1), sg.savgol_filter(c_y_windowed_dy, len(c_y_windowed)-1, 2,1))*fs
        #features["acc"] = np.hypot(sg.savgol_filter(c_x_windowed_dx, len(c_x_windowed_dx)-1, 2,2), sg.savgol_filter(c_y_windowed_dy, len(c_y_windowed)-1, 2,2))*fs**2
        features["vel"] = np.mean(np.hypot(np.diff(c_x_windowed), np.diff(c_y_windowed)) * fs)
        features["acc"] = np.mean(np.hypot(np.diff(np.diff(c_x_windowed)), np.diff(np.diff(c_y_windowed))) * fs ** 2)

        # rayleightest
        angl = np.arctan2(c_y_windowed, c_x_windowed)
        features["rayleightest"] = ast.rayleightest(angl)
        return label,features
