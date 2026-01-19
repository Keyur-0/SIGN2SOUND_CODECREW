import torch
from models.model import SignLSTM

def test_model_forward():
    batch_size = 2
    sequence_length = 30
    feature_dim = 63
    num_classes = 25

    model = SignLSTM(
        input_size=feature_dim,
        hidden_size=256,
        num_classes=num_classes
    )

    dummy_input = torch.randn(batch_size, sequence_length, feature_dim)
    output = model(dummy_input)

    assert output.shape == (batch_size, num_classes), \
        f"Expected output shape {(batch_size, num_classes)}, got {output.shape}"

    print("✅ Model forward pass test passed.")

if __name__ == "__main__":
    test_model_forward()
