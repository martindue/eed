import os
import glob
import time

import astropy.stats as ast
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.signal as sg
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.io import arff
from pathlib import Path
import shutil


x = os.path.dirname(__file__)
print(x)
from ...utils.helpers import vcorrcoef_rolling, loadmat


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

def genData(N, slope, genFun=lambda x: np.random.rand(1,x)):
    data = np.concatenate((genFun(N),genFun(N)),axis=0)
    data = signalSloper(data,slope)
    return data

def signalSloper(x, slope):
    # Ensure x is a 2D array
    x = np.atleast_2d(x)

    # Check if input is a column vector
    qTrans = x.shape[1] == 1
    if qTrans:
        x = x.T

    # Get dimensions
    r, N = x.shape
    numUniquePts = (N + 1) // 2

    # Create index array
    idx = np.arange(1, numUniquePts + 1)

    # Perform FFT along the second dimension
    X = np.fft.fft(x, axis=1)
    #X = X[:, :numUniquePts] * np.concatenate([np.zeros((r, 1)), (idx[:-1] ** (-slope / 2)).reshape(1, -1)], axis=1)
    X = X[:, :numUniquePts] * np.concatenate([ np.zeros((r, 1)),  np.tile(idx[:-1]**(-slope/2),(r,1))], axis = 1)
    # Handle even and odd N
    if N % 2 == 0:  # Even N includes Nyquist point
        X = np.concatenate([X, np.conj(X[:, -2:0:-1])], axis=1)
    else:  # Odd N excludes Nyquist point
        X = np.concatenate([X, np.conj(X[:, -1:0:-1])], axis=1)

    # Perform inverse FFT
    y = np.real(np.fft.ifft(X, axis=1))

    # Subtract mean and divide by standard deviation along the second dimension
    y -= np.mean(y, axis=1, keepdims=True)
    y /= np.std(y, axis=1, keepdims=True)

    # Transpose if originally a column vector
    if qTrans:
        y = y.T

    # ugly fix for the fact that the function sometimes generates two points less than requested
    if y.shape[1] < N:
        ynew = np.zeros((2, N))
        ynew[0] = np.concatenate([y[0], np.array([y[0][-1], y[0][-1]])])
        ynew[1] = np.concatenate([y[1], np.array([y[1][-1], y[1][-1]])])
        y = ynew

    return y

def calcRMSSTD(data):
    RMS = np.sqrt(np.mean(np.sum(np.diff(data, axis=1)**2, axis=0)))
    STD = np.sqrt(np.var(data[0,:]) +  np.var(data[1,:]))
    return RMS, STD
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
        savgol_filter_window: int = 50,
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
        self.window_size_time = long_window_size
        self.window_size_dir_time = window_size_dir
        self.window_size_vel_time = window_size_vel
        self.window_size_samples = None
        self.window_size_dir_samples = None
        self.window_size_vel_samples = None
        self.presaved_features = None
        self.file_names = []
        self.sklearn = sklearn
        self.training_datasets = training_datasets
        self.debugMode = debugMode
        self.fs = None
        self.savgol_filter_window_time = savgol_filter_window

        self.load_raw_data()
        #self.calculate_window_sizes()
        #self.setup_window_indices(long_window_size, window_size_vel, window_size_dir)

        df_concat = None
        for i, df in enumerate(self.data_df_list):
            self.data = np.stack((df["x"], df["y"]), axis=-1)
            self.interpolate_nan_values()
            self.fs = self.calculate_fs(df)
            self.data_df = df
            self.calculate_window_sizes()

            features_df = self.extract_features(self.data[:,0], self.data[:,1])
            if df_concat is None:
                    df_concat = features_df
            else:
                df_concat = pd.concat((df_concat, features_df))

        self.presaved_features = df_concat

    def flip_heading(self,log, col):
        # Flip WCS for selected recordings: applies for Simulator SEP logs
        mask = log[col].values < 0
        log.loc[mask, col] = np.pi + log.loc[mask, col]
        mask = np.logical_or(mask, log[col].values == 0)
        log.loc[~mask, col] = -(np.pi - log.loc[~mask, col])
        return log
    
    def get_load_dirs(self):
        base_path = os.path.join(self.data_dir, "raw")
        
        if self.split == "train":
            if self.training_datasets is None:
                return []

            load_dirs = []
            datasets = {
                "SE_recorded": "train_data/SE_recorded",
                "lund2013": "train_data/lund2013",
                "hollywood2_em": "train_data/hollywood2_em",
                "synthetic_data": "train_data/synthetic_data",
                "lookAtPoint": "train_data/lookAtPoint"
            }

            for dataset, sub_path in datasets.items():
                if dataset in self.training_datasets:
                    load_dirs.append(os.path.join(base_path, sub_path))
                    print(f"Using {dataset.replace('_', ' ').title()} dataset")

        elif self.split == "val":
            load_dirs = [os.path.join(base_path, "val_data")]
        elif self.split == "test":
            load_dirs = [os.path.join(base_path, "test_data")]
        else:
            raise ValueError("Invalid split")

        return load_dirs


        
    def load_raw_data(self):

        self.load_dirs = self.get_load_dirs()
        
        df_list = []

        file_names = []
        file_idx = 0
        for load_dir in self.load_dirs:
            appended_df = None
            if os.path.exists(load_dir) and os.listdir(load_dir):
                file_list = []
                for root, dirs, files in os.walk(load_dir):
                    for file in files:
                        file_list.append(os.path.join(root, file))
                files_to_load = [f for f in file_list if f.endswith(".npy") or f.endswith(".csv") or f.endswith(".arff") or f.endswith(".mat")]
                #file_list = os.listdir(load_dir)
                file_names.extend([os.path.basename(file) for file in files_to_load])
            
            for file_name in tqdm(files_to_load, "Loading files..."):
                file_path = os.path.join(load_dir, file_name)
                if file_name.endswith(".npy"):
                    df = pd.DataFrame(np.load(file_path))
                    if "lookAtPoint" in file_path and 1337 in self.noise_levels:
                        print("len df: ", len(df))
                        noise = genData(len(df), 0.33)
                        print("Noise shape: ", noise.shape)
                        magnitude = 1.5
                        rms, std = calcRMSSTD(noise)
                        print("RMS: ", rms, "STD: ", std)
                        print("RMS/STD: ", rms/std)
                        noise = noise*magnitude/np.hypot(rms, std)
                        print("RMS: ",rms, "STD: ", std)
                        df["x"] = df["x"] + noise[0]
                        df["y"] = df["y"] + noise[1]

                elif file_name.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif file_name.endswith(".arff"):
                    data = arff.loadarff(file_path)
                    df = pd.DataFrame(data[0])
                    df["status"] = True
                    translation_dict = {b'FIX': 1, b'SACCADE': 2, b'SP': 1, b'UNKNOWN': 0, b'NOISE': 0}
                    df["evt"] = df["handlabeller_final"].map(translation_dict)
                    trackloss = np.all(df[["x", "y"]] == 0, axis=1)
                    df["status"] = ~trackloss & (df["evt"]!=0)

                    df = df.drop(columns=["handlabeller_final", "handlabeller_1","confidence"])
                    df["x"] = np.radians(df["x"])
                    df = self.flip_heading(df, 'x')
                    df["x"] = np.degrees(df["x"])
                    df = df.rename(columns={"time":"t"})
                    df["t"] = df["t"]/1e6
                elif file_name.endswith(".mat"):
                    mat = loadmat(file_path)["ETdata"]
                    data = mat["pos"]
                    
                    # read meta data
                    screen_w, screen_h = mat["screenRes"]
                    fs = mat["sampFreq"]

                    # parse data
                    timestamps = data[:, 0].astype(np.float64)
                    if np.all(np.isfinite(timestamps)):
                        mask = timestamps == 0
                        timestamps = timestamps[~mask]
                        timestamps = (timestamps - timestamps[0]) / 1e6
                    else:
                        # some files do not have timestamps. Construct based on sampling rate
                        timestamps = np.arange(len(timestamps)) / fs
                        mask = np.zeros(len(timestamps), dtype=bool)

                    x, y, evt = data[~mask, 3:].T
                    
                    LUND2013_EVENT_MAPPING = {
                        1: 1,  # fixation
                        2: 2,  # saccade
                        3: 1,  # PSO
                        4: 1,  # Pursuit
                        5: 5,  # Blink
                        6: 0,  # undefinded
                    }
                    LUND2013_TRACKLOSS = 0
                    LUND2013_TOLLERANCE = 100
                    status = (
                                np.logical_or(x == LUND2013_TRACKLOSS, y == LUND2013_TRACKLOSS),
                                np.logical_or(x < -LUND2013_TOLLERANCE, x > screen_w + LUND2013_TOLLERANCE),
                                np.logical_or(y < -LUND2013_TOLLERANCE, y > screen_h + LUND2013_TOLLERANCE),
                            )
                    status = np.any(status, axis=0)
                    df = pd.DataFrame({"t": timestamps, "x": x, "y": y, "status": ~status, "evt": evt})
                    df.replace({"evt": LUND2013_EVENT_MAPPING}, inplace=True)

                
                else:
                    raise ValueError("Unsupported file format")
                
                df["file_index"] = file_idx
                file_idx += 1
                if appended_df is None:
                    appended_df = df
                else:
                    appended_df = pd.concat((appended_df, df))
            df_list.append(appended_df)

        self.data_df_list = df_list
        #self.data_df = appended_df
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

    def calculate_window_sizes(self):

        self.window_size_samples = max(int(self.window_size_time/1000 * self.fs),3)
        self.window_size_dir_samples = max(int(self.window_size_dir_time/1000 * self.fs),3)
        self.window_size_vel_samples = max(int(self.window_size_vel_time/1000 * self.fs),3)

        self.savgol_filter_window_samples = max(int(self.savgol_filter_window_time/1000 * self.fs),4)
    
        print("Window size samples: ", self.window_size_samples,)
        print("Window size dir samples: ", self.window_size_dir_samples)
        print("Window size vel samples: ", self.window_size_vel_samples)
        print("Savgol filter window samples: ", self.savgol_filter_window_samples)

    def calculate_fs(self, df, time_column='t'):

        # Ensure the time column is in datetime format
        #df[time_column] = pd.to_datetime(df[time_column])
        temp_df = pd.DataFrame()
        sample_rate_list = []
        for timegrp in df.groupby('file_index')["t"]:
            temp_df = pd.concat([temp_df, timegrp[1].diff()])
            temp_df = temp_df.dropna()
            sample_rate = round(1/temp_df.mean()).values[0]
            sample_rate_list.append(sample_rate)
        if len(sample_rate_list) > 1:
            assert max(np.diff(sample_rate_list)) < 50, "Sample rate varies too much between files in the same directory"

        sample_rate = np.mean(sample_rate_list)
        print(f'Sample rate: {sample_rate} Hz')


        #temp_df['time_diff'] = df.groupby('file_index')[time_column].diff()

        # Remove the first row because its time_diff will be NaT (Not a Time)
        #temp_df = temp_df.dropna(subset=['time_diff'])

        # Calculate the most common interval (mode)
        #sample_rate = round(1/temp_df['time_diff'].mode()[0])

        # Print the sample rate
        #print(f'Sample rate: {sample_rate} Hz')
        return sample_rate


    def setup_window_indices(self, long_window_size: int, window_size_vel: int, window_size_dir: int):
        (
            self.pre_window_indices,
            self.center_window_indices,
            self.post_window_indices,
        ) = get_window_indices(
            self.data_df, {"window_size": long_window_size, "window_resolution": 1}
        )


    def __len__(self):
        return 1 #len(self.presaved_features)  # number of windows

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
        ## for later epochs, perform online augmentation
        #if not self.sklearn and self.trainer.current_epoch >= self.max_presaved_epoch:
        #    features_df = self.online_augmentation(idx)
        #    return {
        #        "features": torch.squeeze(
        #            torch.tensor(
        #                features_df.loc[:, features_df.columns != "label"].values,
        #                dtype=torch.float32,
        #            )
        #        ),
        #        "label": torch.squeeze(
        #            torch.tensor(features_df.loc[:, "label"], dtype=torch.long)
        #        ),
        #        "t": self.data_df.loc[idx, "t"],
        #        "x": self.data_df.loc[idx, "x"],
        #        "y": self.data_df.loc[idx, "y"],
        #        "status": self.data_df.loc[idx, "status"],
        #        "file_index": self.data_df.loc[idx, "file_index"],
        #        "file_name": self.data_df.loc[idx, "file_name"],
        #    }
        #else:
        #    return {
        #        "features": torch.tensor(
        #            self.presaved_features.iloc[idx, 7:],
        #            dtype=torch.float32,
        #        ),
        #        "label": torch.tensor(self.presaved_features.iat[idx, 4], dtype=torch.long),
        #        "t": self.presaved_features.iat[idx, 0],
        #        "x": self.presaved_features.iat[idx, 1],
        #        "y": self.presaved_features.iat[idx, 2],
        #        "status": self.presaved_features.iat[idx, 3],
        #        "file_name": self.file_names[self.presaved_features.iat[idx, 5]],
        #        "file_index": self.presaved_features.iat[idx, 5],
        #    }
        #features =  torch.tensor(
        #        self.presaved_features.iloc[:, 7:].values,
        #        dtype=torch.float32
        #    )
        df = { "label":self.presaved_features.iloc[:,4].values.squeeze(), #torch.tensor(self.presaved_features.iloc[:, 4].values, dtype=torch.long).squeeze(),
            "t": self.presaved_features.iloc[:, 0].values.squeeze(),
            "x": self.presaved_features.iloc[:, 1].values.squeeze(),
            "y": self.presaved_features.iloc[:, 2].values.squeeze(),
            "status": self.presaved_features.iloc[:, 3].values.squeeze(),
            #"file_name": [self.file_names[i] for i in self.presaved_features.iloc[:, 5].values],
            "file_index": self.presaved_features.iloc[:, 5].values.squeeze(),
        }
        #return df, features
        #df = { "label":[self.presaved_features["label"].to_numpy()], #torch.tensor(self.presaved_features.iloc[:, 4].values, dtype=torch.long).squeeze(),
        #    "t": [self.presaved_features["t"].to_numpy()],
        #    "x": [self.presaved_features["x"].to_numpy()],
        #    "y": [self.presaved_features["y"].to_numpy()],
        #    "status": [self.presaved_features["status"].to_numpy()],
        #    #"file_name": self.presaved_features["file_name"].values.tolist(),
        #    "file_index": [self.presaved_features["file_index"].to_numpy()],
        #}
        #label_list = self.presaved_features["label"].to_list()
        #t_list = self.presaved_features["t"].to_list()
        #x_list = self.presaved_features["x"].to_list()
        #y_list = self.presaved_features["y"].to_list()
        #status_list = self.presaved_features["status"].to_list()
        #file_index_list = self.presaved_features["file_index"].to_list()
        #data_list = [label_list, t_list, x_list, y_list, status_list, file_index_list]
        features =  torch.tensor(
                self.presaved_features.iloc[:, 7:].values,
                dtype=torch.float32
            )
        file_name_list = [self.file_names[i] for i in self.presaved_features.iloc[:, 5].values]
        return df, features, file_name_list
        
    def extract_features(self, x,y):
        x_df = pd.DataFrame(x)
        y_df = pd.DataFrame(y)
        feature_dict = {}
        # differentiate the data to get velocity and acceleration
        vel_data = np.hypot(
            sg.savgol_filter(x, self.savgol_filter_window_samples, 3, 1, axis=0),
            sg.savgol_filter(y, self.savgol_filter_window_samples, 3, 1, axis=0)
        )*self.fs
        acc_data = np.hypot(
            sg.savgol_filter(x, self.savgol_filter_window_samples, 3, 2, axis=0),
            sg.savgol_filter(y, self.savgol_filter_window_samples, 3, 2, axis=0)
        )*self.fs**2
        # take a moving average of the acceleration, using pandas
        acc_data_averaged = pd.DataFrame(acc_data).rolling(window=self.window_size_vel_samples, center=True).mean().bfill().ffill().values
        
        feature_dict["t"] = self.data_df["t"].values
        feature_dict["x"] = self.data_df["x"].values
        feature_dict["y"] = self.data_df["y"].values
        feature_dict["status"] = self.data_df["status"].values
        feature_dict["label"] = self.data_df["evt"] #self.labels.values
        feature_dict["file_index"] = self.data_df["file_index"].values
        feature_dict["file_name"] = [self.file_names[i] for i in self.data_df["file_index"].values]

        feature_dict["vel"] = vel_data
        feature_dict["acc"] = acc_data
        feature_dict["acc_averaged"] = acc_data_averaged.flatten()

        std_x = x_df.rolling(window=self.window_size_samples, center=True).std().bfill().ffill().values.flatten()
        std_y = y_df.rolling(window=self.window_size_samples, center=True).std().bfill().ffill().values.flatten()
        std = np.hypot(std_x, std_y)
        feature_dict["std"] = std
        feature_dict["std-diff"] = np.abs(np.roll(std, -(self.window_size_samples-1)//2) - np.roll(std, (self.window_size_samples-1)//2))

        mean_diff_x = x_df.shift(-self.window_size_samples//2).rolling(window=self.window_size_samples, center=True).mean() - x_df.shift(self.window_size_samples//2).rolling(window=self.window_size_samples, center=False).mean()
        mean_diff_y = y_df.shift(-self.window_size_samples//2).rolling(window=self.window_size_samples, center=True).mean() - y_df.shift(self.window_size_samples//2).rolling(window=self.window_size_samples, center=False).mean() 
        mean_diff = np.hypot(mean_diff_x.ffill().bfill().values.flatten(), mean_diff_y.ffill().bfill().values.flatten())

        feature_dict["mean-diff"] = mean_diff  

        med_diff_x = x_df.shift(-self.window_size_samples//2).rolling(window=self.window_size_samples, center=True).median() - x_df.shift(self.window_size_samples//2).rolling(window=self.window_size_samples, center=True).median()
        med_diff_y = y_df.shift(-self.window_size_samples//2).rolling(window=self.window_size_samples, center=True).median() - y_df.shift(self.window_size_samples//2).rolling(window=self.window_size_samples, center=True).median()
        med_diff = np.hypot(med_diff_x.ffill().bfill().values.flatten(), med_diff_y.ffill().bfill().values.flatten())

        feature_dict["med-diff"] = med_diff



        # bcea
        P = 0.68
        k = np.log(1 / (1 - P))
        rho = vcorrcoef_rolling(x_df, y_df,self.window_size_samples)
        feature_dict["bcea"] = 2*k*np.pi*std_x*std_y* np.sqrt(1-np.power(rho,2)).values.flatten()
        
        #bcea_diff
        feature_dict['bcea-diff-directional'] = np.roll(feature_dict['bcea'], -(self.window_size_samples-1)//2) - \
                    np.roll(feature_dict['bcea'], (self.window_size_samples-1)//2)
        feature_dict['bcea-diff'] = np.abs(feature_dict['bcea-diff-directional'])
        del feature_dict['bcea-diff-directional']


        # RMS
        rms_x = np.sqrt(np.square(x_df).rolling(window=self.window_size_samples, center=True).mean().bfill().ffill().values.flatten())
        rms_y = np.sqrt(np.square(y_df).rolling(window=self.window_size_samples, center=True).mean().bfill().ffill().values.flatten())
        feature_dict["rms"] = np.hypot(rms_x, rms_y)

        feature_dict["rms-diff"] = np.roll(feature_dict["rms"], -(self.window_size_samples-1)//2) - np.roll(feature_dict["rms"], (self.window_size_samples-1)//2)

        # dispersion
        x_range = x_df.rolling(window=self.window_size_samples, center=True).max().bfill().ffill().values.flatten() - x_df.rolling(window=self.window_size_samples, center=True).min().bfill().ffill().values.flatten()
        y_range = y_df.rolling(window=self.window_size_samples, center=True).max().bfill().ffill().values.flatten() - y_df.rolling(window=self.window_size_samples, center=True).min().bfill().ffill().values.flatten()
        feature_dict["disp"] = x_range + y_range

        # rayleightest
        #angl = np.arctan2(y_df.rolling(window=self.window_size_dir_samples, center=True).mean().bfill().ffill().values.flatten(), x_df.rolling(window=self.window_size_dir_samples, center=True).mean().bfill().ffill().values.flatten())
        #feature_dict["rayleightest"] = ast.rayleightest(angl)

        features_df = pd.DataFrame.from_dict(feature_dict)

        return features_df

 