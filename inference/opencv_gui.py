import cv2
import numpy as np
import queue
import sys
from collections import deque

from speech_to_text.vosk_stt import VoskSTT
from features.hand_landmarks import HandLandmarkExtractor
from inference.infer import load_model, predict_sign

# ---------------- CONFIG ----------------
WINDOW_NAME = "SIGN2SOUND"
SEQUENCE_LENGTH = 30

# Temporal stability
LOCK_THRESHOLD = 8        # frames to lock sign
UNLOCK_THRESHOLD = 5     # frames with no hand to reset

# VOSK STT
MODEL_PATH = "speech_to_text/models/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 48000
MIC_DEVICE_INDEX = None

# ---------------- GLOBAL STATE ----------------
running = True
listening = False
current_text = ""

sequence_buffer = []

live_sign = ""
locked_sign = ""
status_text = "Waiting..."

# Temporal lock state
candidate_sign = None
candidate_count = 0
no_hand_count = 0

# Audio queue (required by Vosk)
audio_queue = queue.Queue()


# ---------------- AUDIO CALLBACK ----------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))


# ---------------- MAIN ----------------
def main():
    global running, listening, current_text
    global live_sign, locked_sign, status_text
    global candidate_sign, candidate_count, no_hand_count

    # Init STT
    stt = VoskSTT(
        model_path=MODEL_PATH,
        sample_rate=SAMPLE_RATE,
        device=MIC_DEVICE_INDEX
    )
    stt.start()
    
    COOLDOWN_FRAMES = 12   # ~0.4 sec at 30 FPS
    cooldown_count = 0
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
        ret, cam_frame = cam.read()
        if not ret:
            continue

        # Preprocess frame
        cam_frame = cv2.flip(cam_frame, 1)
        cam_small = cv2.resize(cam_frame, (320, 240))

        # ---------------- FEATURE EXTRACTION ----------------
        features = extractor.extract(cam_small)

        # ---- NO HAND DETECTION ----
        if np.count_nonzero(features) < 20:
            no_hand_count += 1
        else:
            no_hand_count = 0
            sequence_buffer.append(features)

        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)

        # ---- HARD RESET if hand seen gone ----
        if no_hand_count >= UNLOCK_THRESHOLD:
            sequence_buffer.clear()
            live_sign = ""
            locked_sign = ""
            candidate_sign = None
            candidate_count = 0
            status_text = "Waiting..."
            continue

        # ---------------- PREDICTION ----------------
        if len(sequence_buffer) < SEQUENCE_LENGTH:
            status_text = "Collecting..."
            live_sign = ""
        else:
            seq_np = np.array(sequence_buffer, dtype=np.float32)
            live_sign = predict_sign(seq_np, model)

            # -------- TEMPORAL LOCK --------
            if live_sign == candidate_sign:
                candidate_count += 1
            else:
                candidate_sign = live_sign
                candidate_count = 1

            if candidate_count >= LOCK_THRESHOLD:
                locked_sign = candidate_sign
                status_text = "Sign Stable"
            else:
                status_text = "Predicting..."

        # ---------------- GUI ----------------
        FRAME_W, FRAME_H = 720, 420
        frame = np.ones((FRAME_H, FRAME_W, 3), dtype=np.uint8) * 245

        # Webcam placement
        CAM_W, CAM_H = 280, 210
        CAM_X, CAM_Y = 400, 140
        cam_disp = cv2.resize(cam_small, (CAM_W, CAM_H))
        frame[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cam_disp

        cv2.rectangle(
            frame,
            (CAM_X - 2, CAM_Y - 2),
            (CAM_X + CAM_W + 2, CAM_Y + CAM_H + 2),
            (0, 0, 0),
            2
        )

        # ---------------- TEXT ----------------
        cv2.putText(frame, "SIGN2SOUND",
                    (40, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)

        cv2.putText(frame, "SIGN:",
                    (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        cv2.putText(frame, locked_sign if locked_sign else "-",
                    (150, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(frame, f"Predicting: {live_sign}",
                    (40, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 2)

        cv2.putText(frame, "Speech:",
                    (40, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        cv2.putText(frame, f"\"{current_text[:32]}\"",
                    (40, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

        cv2.putText(frame, "Status:",
                    (40, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        cv2.putText(frame, status_text,
                    (150, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)

        # ---------------- CONTROLS ----------------
        base_y = FRAME_H - 120
        cv2.putText(frame, "Controls:",
                    (20, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.putText(frame, "S = STT Start   E = STT Stop",
                    (20, base_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 2)

        cv2.putText(frame, "Q = Quit",
                    (20, base_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 2)

        cv2.imshow(WINDOW_NAME, frame)

        # ---------------- KEYS ----------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running = False
        elif key == ord("s"):
            listening = True
            stt.set_listening(True)
        elif key == ord("e"):
            listening = False
            stt.set_listening(False)

        if listening:
            current_text = stt.get_text()

    # Cleanup
    stt.stop()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()