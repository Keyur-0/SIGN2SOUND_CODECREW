import torch
from models.model import SignLSTM

def test_model_forward():
    model = SignLSTM(input_size=63, hidden_size=256, num_classes=25)
    x = torch.randn(1, 30, 63)  # (batch, seq_len, features)

    y = model(x)

    assert y.shape == (1, 25)

