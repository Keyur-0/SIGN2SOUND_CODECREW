import torch
import torch.nn as nn

# Define the LSTM-based model for sign language recognition
class SignLSTM(nn.Module):
    # Initialize the LSTM model
    def __init__(self, input_size=63, hidden_size=128, num_classes=26):
        super(SignLSTM, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        # Final fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
    # Define the forward pass
    def forward(self, x):
        # x: (batch_size, 30, 63)
        lstm_out, _ = self.lstm(x)

        # Take last time step
        last_hidden = lstm_out[:, -1, :]

        out = self.fc(last_hidden)
        return out
