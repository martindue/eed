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

    def __init__(self, data_dir):
        self.data_dir = data_dir+"/raw"

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
        self.x = appended_array['x']
        self.y = appended_array['y']
        self.status = appended_array['status']
        self.evt = appended_array['evt']
        self.input = np.stack((self.x, self.y), axis=-1)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        itemDict = {"input": self.input[index], "evt": self.evt[index]}
        return itemDict