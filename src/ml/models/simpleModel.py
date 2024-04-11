import torch
import torch.nn.functional as F


class TorchModel(torch.nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_classes: int = 6):
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
