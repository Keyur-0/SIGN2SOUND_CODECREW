import torch
import numpy as np
from models.model import SignLSTM

# ---------- Configuration ----------
MODEL_PATH = "checkpoints/sign_lstm.pth"
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 63
DEVICE = torch.device("cpu")

# MUST match training labels (25 classes)
IDX_TO_CLASS = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","S","T",
    "U","V","W","X","Y","Z"
]

# ---------- Load model ----------
def load_model():
    model = SignLSTM(
        input_size=FEATURES_PER_FRAME,
        hidden_size=256,          # ✅ MATCH TRAINING
        num_classes=len(IDX_TO_CLASS)  # ✅ 25
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ---------- Inference ----------
def predict_sign(sequence: np.ndarray, model):
    """
    sequence: numpy array of shape (30, 63)
    returns: predicted class label (e.g. "A")
    """
    assert sequence.shape == (SEQUENCE_LENGTH, FEATURES_PER_FRAME)

    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        pred_idx = torch.argmax(logits, dim=1).item()

    return IDX_TO_CLASS[pred_idx]

# ---------- Test ----------
if __name__ == "__main__":
    model = load_model()
    dummy_sequence = np.random.rand(30, 63).astype(np.float32)
    prediction = predict_sign(dummy_sequence, model)
    print("Predicted sign:", prediction)
