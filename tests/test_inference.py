import numpy as np
from inference.infer import predict_sign

class DummyModel:
    def __call__(self, x):
        import torch
        return torch.randn(1, 25)

def test_predict_sign():
    dummy_model = DummyModel()
    sequence = np.random.rand(30, 63).astype("float32")

    label = predict_sign(sequence, dummy_model)

    assert isinstance(label, str)
