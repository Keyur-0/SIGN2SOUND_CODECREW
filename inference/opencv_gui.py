import cv2
import numpy as np
import queue
import sys
from collections import deque, Counter

from speech_to_text.vosk_stt import VoskSTT
from features.hand_landmarks import HandLandmarkExtractor
from inference.infer import load_model, predict_sign

# ---------------- CONFIG ----------------
WINDOW_NAME = "SIGN2SOUND"
SEQUENCE_LENGTH = 30
SMOOTHING_WINDOW = 9

# VOSK STT
MODEL_PATH = "speech_to_text/models/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 48000
MIC_DEVICE_INDEX = None

# ---------------- GLOBAL STATE ----------------
running = True
listening = False
current_text = ""

sequence_buffer = []
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

live_sign = ""
stable_sign = ""
status_text = "Waiting..."

# Recording (optional / future)
RECORDING = False
current_label = None
recorded_sequences = []

audio_queue = queue.Queue()


# ---------------- AUDIO CALLBACK ----------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))


# ---------------- MAIN ----------------
def main():
    global running, listening, current_text
    global live_sign, stable_sign, status_text
    global RECORDING, current_label

    # Init STT
    stt = VoskSTT(
        model_path=MODEL_PATH,
        sample_rate=SAMPLE_RATE,
        device=MIC_DEVICE_INDEX
    )
    stt.start()

    # Init Sign model + extractor
    extractor = HandLandmarkExtractor()
    model = load_model()

    # Open webcam
    print("[INFO] Opening webcam...")
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Could not open webcam")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while running:
        finger_states = None

        ret, cam_frame = cam.read()
        if not ret:
            continue

        # Preprocess camera frame
        cam_frame = cv2.flip(cam_frame, 1)
        cam_frame_small = cv2.resize(cam_frame, (320, 240))

        # -------- Feature extraction (126D) --------
        features = extractor.extract(cam_frame_small)
        # sequence_buffer.append(features)
        if np.count_nonzero(features) > 20:  # hand actually detected
            sequence_buffer.append(features)
        else:
            # bad frame → do NOT add noise
            sequence_buffer.clear()
            prediction_buffer.clear()
            live_sign = ""
            stable_sign = ""
            status_text = "Waiting..."

        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)

        # -------- Prediction logic --------
        if len(sequence_buffer) < SEQUENCE_LENGTH:
            status_text = "Collecting..."
            live_sign = ""
            stable_sign = ""
            prediction_buffer.clear()
        else:
            seq_np = np.array(sequence_buffer, dtype=np.float32)
            live_sign = predict_sign(seq_np, model)
            finger_states = extractor.last_finger_states
            prediction_buffer.append(live_sign)

            most_common, count = Counter(prediction_buffer).most_common(1)[0]
            if count >= 6:
                stable_sign = most_common
                status_text = "Sign Stable"
            else:
                stable_sign = ""
                status_text = "Predicting..."

        if finger_states is not None and live_sign in ["B", "D", "G"]:
            fs = finger_states.tolist()

            # ISL B: all fingers extended, thumb folded
            if fs == [0, 1, 1, 1, 1]:
                live_sign = "B"

            # ISL D: only index finger extended
            elif fs == [0, 1, 0, 0, 0]:
                live_sign = "D"

        # -------- Optional recording --------
        if RECORDING and len(sequence_buffer) == SEQUENCE_LENGTH:
            recorded_sequences.append(
                (current_label, np.array(sequence_buffer))
            )
            print(f"[RECORDED] Label: {current_label}, Shape: (30, 126)")
            RECORDING = False

        # -------- GUI CANVAS --------
        FRAME_W, FRAME_H = 720, 420
        frame = np.ones((FRAME_H, FRAME_W, 3), dtype=np.uint8) * 245

        # Webcam placement
        CAM_W, CAM_H = 280, 210
        CAM_X, CAM_Y = 400, 140
        cam_small = cv2.resize(cam_frame_small, (CAM_W, CAM_H))
        frame[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cam_small

        cv2.rectangle(
            frame,
            (CAM_X - 2, CAM_Y - 2),
            (CAM_X + CAM_W + 2, CAM_Y + CAM_H + 2),
            (0, 0, 0),
            2
        )

        # -------- TEXT --------
        cv2.putText(frame, "SIGN2SOUND",
                    (40, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)

        cv2.putText(frame, "SIGN:",
                    (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        cv2.putText(frame, stable_sign if stable_sign else "-",
                    (150, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(frame, f"Predicting: {live_sign}",
                    (40, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 2)

        cv2.putText(frame, "Speech:",
                    (40, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        speech_display = current_text[:32]
        cv2.putText(frame, f"\"{speech_display}\"",
                    (40, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

        cv2.putText(frame, "Status:",
                    (40, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        cv2.putText(frame, status_text,
                    (150, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)

        # -------- CONTROLS --------
        base_y = FRAME_H - 120
        cv2.putText(frame, "Controls:",
                    (20, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.putText(frame, "S = STT Start   E = STT Stop",
                    (20, base_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 2)

        cv2.putText(frame, "R = Record   Q = Quit",
                    (20, base_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 2)

        cv2.imshow(WINDOW_NAME, frame)

        # -------- KEYS --------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running = False
        elif key == ord("s"):
            listening = True
            stt.set_listening(True)
        elif key == ord("e"):
            listening = False
            stt.set_listening(False)
        elif key == ord("r"):
            RECORDING = True
            current_label = "test_gesture"
            sequence_buffer.clear()

        if listening:
            current_text = stt.get_text()

    # Cleanup
    stt.stop()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    