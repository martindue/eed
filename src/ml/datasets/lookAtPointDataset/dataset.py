"""
Replace the following code with your own dataset class.
Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
Example:

from torch.utils.data import Dataset


class Dataset1(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return x[index], y[index]
"""
from torch.utils.data import Dataset
import numpy as np
import os

class LookAtPointDataset(Dataset):

    def __init__(self, data_dir, window_size, stride):
        self.data_dir = data_dir+"/raw"
        self.window_size = window_size
        self.stride = stride

        file_list = os.listdir(self.data_dir)
        print("Files in data directory:", file_list)
        numpy_files = [f for f in file_list if f.endswith('.npy')]
        appended_array = None
        for file_name in numpy_files:
            file_path = os.path.join(self.data_dir, file_name)
            loaded_array = np.load(file_path)
            if appended_array is None:
                appended_array = loaded_array
            else:
                appended_array = np.concatenate((appended_array, loaded_array))
        print("Shape of data:", appended_array.shape)
        self.t = appended_array['t']
        self.x_data = appended_array['x']
        self.y_data = appended_array['y']
        self.status = appended_array['status']
        self.evt = appended_array['evt']
        #self.input = np.stack((self.x, self.y), axis=-1)
        self.data = self.create_dataset()

    def create_dataset(self):
        data = []
        for i in range(0, len(self.x_data) - self.window_size, self.stride):
            window_x = self.x_data[i:i+self.window_size]
            window_y = self.y_data[i:i+self.window_size]
            window_x = np.ascontiguousarray(window_x)
            window_y = np.ascontiguousarray(window_y)

            evt_labels = self.evt[i+self.window_size-1]
            data.append((window_x, window_y, evt_labels))
        return data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        print(f"Index: {index}, Data length: {len(self.data)}")
        if index >= len(self.data):
            raise IndexError("Index out of range")
        window_x, window_y, evt_labels = self.data[index]
        print("window_x.shape: ", window_x.shape)
        print("window_y.shape: ", window_y.shape)
        if index+self.window_size > len(self.status):
            raise IndexError("Index + window size out of range")
        return {
                'input': (window_x, window_y),
                'evt': evt_labels,
                'status': self.status[index:index+self.window_size]
            }
