import torch
import torch.nn as nn

class SignLSTM(nn.Module):
    def __init__(self, input_size=126, hidden_size=256, num_classes=25):
        super(SignLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,        # required for dropout
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, 126)
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # last time-step
        out = self.fc(out)
        return out