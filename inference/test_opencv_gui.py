import cv2
import numpy as np
import time

from features.hand_landmarks import HandLandmarkExtractor
from inference.infer import load_model, predict_sign

SEQUENCE_LENGTH = 30


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    extractor = HandLandmarkExtractor()
    model = load_model()

    sequence = []
    last_pred = ""
    fps_time = time.time()

    print("[INFO] Two-hand test module started")
    print("[INFO] Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract two-hand features (126,)
        features = extractor.extract(frame)
        sequence.append(features)

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        # Predict once buffer is full
        if len(sequence) == SEQUENCE_LENGTH:
            seq_np = np.array(sequence, dtype=np.float32)
            last_pred = predict_sign(seq_np, model)

        # FPS calculation
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()

        # Overlay text
        cv2.putText(
            frame,
            f"Prediction: {last_pred}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Mode: Two-hand test",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

        cv2.imshow("SIGN2SOUND – Two-hand Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()