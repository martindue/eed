import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import os

class LookAtPointDatasetMiddleLabel(Dataset):
    def __init__(self, data_dir: str = "/home/martin/Documents/Exjobb/eed/.data", window_size: int = 250):
        data_dir = data_dir+"/raw"
        file_list = os.listdir(data_dir)
        print("Files in data directory:", file_list)
        numpy_files = [f for f in file_list if f.endswith('.npy')]
        appended_array = None
        for file_name in numpy_files:
            file_path = os.path.join(data_dir, file_name)
            loaded_array = np.load(file_path)
            if appended_array is None:
                appended_array = loaded_array
            else:
                appended_array = np.concatenate((appended_array, loaded_array))
        print("Shape of data:", appended_array.shape)
        t = appended_array['t']
        x_data = appended_array['x']
        y_data = appended_array['y']
        status = appended_array['status']
        evt = appended_array['evt']
        labels = evt

        data = np.stack((x_data, y_data), axis=-1)

        # Interpolate NaN values for each column separately. Placeholder.
        for i in range(data.shape[1]):
            nan_indices = np.isnan(data[:, i])
            if np.any(nan_indices):
                data[:, i][nan_indices] = np.interp(np.flatnonzero(nan_indices), np.flatnonzero(~nan_indices), data[:, i][~nan_indices])

        self.data = data
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        window_data = self.data[idx:idx+self.window_size]
        middlePoint = idx + self.window_size//2
        label = self.labels[middlePoint]  # The label corresponds to the middle element in the window
        return window_data, label