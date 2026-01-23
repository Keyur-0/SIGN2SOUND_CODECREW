import mediapipe as mp
import cv2
import numpy as np

import json
import queue
import sys

from speech_to_text.vosk_stt import VoskSTT

from inference.infer import load_model, predict_sign
from collections import deque, Counter

# GLOBAL VARIABLES
current_sign = ""

# DEBUG VARIABLES
DEBUG_LANDMARKS = False

# PREDICTION SMOOTHING VARIABLES
SMOOTHING_WINDOW = 7
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

live_sign = ""
stable_sign = ""

# MEDIAPIPE VARIABLES FOR LSTMS
SEQUENCE_LENGTH = 30
sequence_buffer = []

# DATA RECORDING VARIABLES
RECORDING = False
current_label = None
recorded_sequences = []

# VOSK STT VARIABLES
listening = False
running = True
MODEL_PATH = "speech_to_text/models/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 48000
MIC_DEVICE_INDEX = None

# OPEN CV GUI VARIABLES
WINDOW_NAME = "SIGN2SOUND"
current_text = ""

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))

def main():
    global running, current_text, listening
    global DEBUG_LANDMARKS, RECORDING, current_label
    global sequence_buffer, recorded_sequences, current_sign
    global stable_sign, live_sign
    stt = VoskSTT(
    model_path=MODEL_PATH,
    sample_rate=SAMPLE_RATE,
    device=MIC_DEVICE_INDEX
    )
    stt.start()

    # Load sign model
    sign_model = load_model()

    # Open webcam
    print("Opening webcam...")
    cam = cv2.VideoCapture(0)

    # MediaPipe Hands (ALLOW BOTH HANDS)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,   # ✅ BOTH HANDS
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while running:
        ret, cam_frame = cam.read()
        if not ret:
            continue
        status_text = "Waiting..."
        cam_frame = cv2.resize(cam_frame, (320, 240))
        cam_frame = cv2.flip(cam_frame, 1)

        rgb_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # 🔥 PROCESS *ALL* DETECTED HANDS
            for hand_landmarks in results.multi_hand_landmarks:

                landmark_vector = []
                for lm in hand_landmarks.landmark:
                    landmark_vector.extend([lm.x, lm.y, lm.z])

                landmark_vector = np.array(landmark_vector, dtype=np.float32)

                sequence_buffer.append(landmark_vector)
                if len(sequence_buffer) > SEQUENCE_LENGTH:
                    sequence_buffer.pop(0)

                if len(sequence_buffer) < SEQUENCE_LENGTH:
                    status_text = "Collecting..."
                    stable_sign = ""
                    prediction_buffer.clear()

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    sequence = np.stack(sequence_buffer)

                    # Live prediction
                    live_sign = predict_sign(sequence, sign_model)
                    prediction_buffer.append(live_sign)

                    # Check stability
                    most_common, count = Counter(prediction_buffer).most_common(1)[0]

                    if count >= 5:   # 5 out of 7
                        stable_sign = most_common
                        status_text = "Sign Stable"
                    else:
                        status_text = "Predicting..."

                    if RECORDING:
                        recorded_sequences.append((current_label, sequence))
                        print(f"[RECORDED] Label: {current_label}, Shape: {sequence.shape}")
                        RECORDING = False
                

                if DEBUG_LANDMARKS:
                    print("Landmark vector shape:", landmark_vector.shape)

                mp_drawing.draw_landmarks(
                    cam_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
        if not results.multi_hand_landmarks:
            sequence_buffer.clear()
            prediction_buffer.clear()
            live_sign = ""
            stable_sign = ""
            status_text = "Waiting..."

        # ---------------- GUI ----------------
        FRAME_W, FRAME_H = 720, 420
        frame = np.ones((FRAME_H, FRAME_W, 3), dtype=np.uint8) * 245


        # Webcam placement (RIGHT SIDE)
        CAM_W, CAM_H = 280, 210
        CAM_X, CAM_Y = 400, 140

        cam_frame = cv2.resize(cam_frame, (CAM_W, CAM_H))
        frame[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = cam_frame

        # Webcam border
        cv2.rectangle(
            frame,
            (CAM_X-2, CAM_Y-2),
            (CAM_X+CAM_W+2, CAM_Y+CAM_H+2),
            (0, 0, 0),
            2
        )

        TITLE_FONT = 1.1
        LABEL_FONT = 0.75
        VALUE_FONT = 0.9
        STATUS_FONT = 0.8

        # Title
        cv2.putText(frame, "SIGN2SOUND",
                    (40, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, TITLE_FONT, (0,0,0), 2)

        # Sign
        cv2.putText(frame, "SIGN:",
                    (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT, (0,0,0), 2)

        # Speech (truncate to avoid overflow)
        MAX_CHARS = 32
        speech_display = current_text[:MAX_CHARS]

        cv2.putText(frame, 'Speech:',
                    (40, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT, (0,0,0), 2)

        cv2.putText(frame, f'"{speech_display}"',
                    (40, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,30,30), 2)

        # Status
        cv2.putText(frame, "Status:",
                    (40, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT, (0,0,0), 2)

        cv2.putText(frame, status_text,
                    (150, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, STATUS_FONT, (0,150,0), 2)
        cv2.putText(frame, stable_sign if stable_sign else "-",
                    (150, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, VALUE_FONT, (0,0,255), 2)

        cv2.putText(frame, f"Predicting: {live_sign}",
                    (40, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,80,80), 2)
        
        # ---------- CONTROLS (BOTTOM LEFT) ----------
        CTRL_X = 20
        CTRL_Y = frame.shape[0] - 120   # distance from bottom

        CTRL_TITLE_FONT = 0.6
        CTRL_TEXT_FONT = 0.5
        CTRL_COLOR_TITLE = (0, 0, 0)
        CTRL_COLOR_TEXT = (60, 60, 60)
        THICKNESS = 2

        cv2.putText(
            frame, "Controls:",
            (CTRL_X, CTRL_Y),
            cv2.FONT_HERSHEY_SIMPLEX,
            CTRL_TITLE_FONT,
            CTRL_COLOR_TITLE,
            THICKNESS
        )

        cv2.putText(
            frame, "Speech → Text :  S = Start   E = Stop",
            (CTRL_X, CTRL_Y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            CTRL_TEXT_FONT,
            CTRL_COLOR_TEXT,
            THICKNESS
        )

        cv2.putText(
            frame, "Record Sign :  R = Record",
            (CTRL_X, CTRL_Y + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            CTRL_TEXT_FONT,
            CTRL_COLOR_TEXT,
            THICKNESS
        )

        cv2.putText(
            frame, "Landmarks :  L",
            (CTRL_X, CTRL_Y + 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            CTRL_TEXT_FONT,
            CTRL_COLOR_TEXT,
            THICKNESS
        )

        cv2.putText(
            frame, "Quit :  Q",
            (CTRL_X, CTRL_Y + 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            CTRL_TEXT_FONT,
            CTRL_COLOR_TEXT,
            THICKNESS
        )


        cv2.imshow(WINDOW_NAME, frame)


        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running = False
            break
        elif key == ord("s"):
            listening = True
            stt.set_listening(True)
        elif key == ord("e"):
            listening = False
            stt.set_listening(False)
        elif key == ord("l"):
            DEBUG_LANDMARKS = not DEBUG_LANDMARKS
        elif key == ord("r"):
            RECORDING = True
            current_label = "test_gesture"
            sequence_buffer.clear()
        elif key == ord("t"):
            RECORDING = False
        # Update current text from STT
        if listening:
            current_text = stt.get_text()

    stt.stop()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
