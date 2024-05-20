import torch
import torch.nn.functional


class TorchModel(torch.nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 64, num_classes: int = 6):
        super(TorchModel, self).__init__()

        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(
            lstm_out[:, x.size(1) // 2, :]
        )  # Taking the output of the middle time step
        return output

    # def training_step(self, batch, batch_idx):
    #    x, y = batch
    #    print("x.shape: ", x.shape)
    #    print("y.shape: ", y.shape)
    #    y_hat = self(x)
    #    loss = torch.nn.functional.cross_entropy(y_hat, y)
    #    self.log('train_loss', loss)
    #    return loss
