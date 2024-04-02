"""
Replace the following code with your own nn.Module class.
Reference: https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
Example:

import torch

class TorchModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
"""
import torch
import torch.nn.functional as F


class TorchModel(torch.nn.Module):

    def __init__(self):
        super(TorchModel, self).__init__()

        self.linear1 = torch.nn.LazyLinear(100)

        self.linear2 = torch.nn.Linear(100, 50)
        self.linear3 = torch.nn.Linear(50, 6)
        self.softmax = torch.nn.Softmax()


    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.softmax(x)
        return x
