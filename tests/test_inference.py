import numpy as np
from inference.infer import load_model, predict_sign

def test_inference_pipeline():
    model = load_model()

    # Dummy landmark sequence: (30 frames × 63 features)
    dummy_sequence = np.random.rand(30, 63).astype("float32")

    prediction = predict_sign(dummy_sequence, model)

    assert prediction is not None, "Prediction returned None"
    assert isinstance(prediction, str), "Prediction should be a class label (string)"

    print("✅ Inference pipeline test passed.")

if __name__ == "__main__":
    test_inference_pipeline()
