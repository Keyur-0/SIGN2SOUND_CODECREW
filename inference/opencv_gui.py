# TODO: Test if the OpenCV GUI integration with Vosk STT works as intended
import mediapipe as mp
import cv2
import numpy as np

import json
import queue
import sys

import sounddevice as sd
from vosk import KaldiRecognizer, Model
from inference.infer import load_model, predict_sign

# GLOBAL VARIABLES
current_sign = ""

# DEBUG VARIABLES
DEBUG_LANDMARKS = False

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

    # Load sign model
    sign_model = load_model()

    # Load Vosk model
    print("Loading Vosk model...")
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)

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

    # Start audio stream
    print("Starting audio stream...")
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        device=MIC_DEVICE_INDEX,
        dtype="int16",
        channels=1,
        callback=audio_callback
    )
    stream.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while running:
        ret, cam_frame = cam.read()
        if not ret:
            continue

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

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    sequence = np.stack(sequence_buffer)
                    current_sign = predict_sign(sequence, sign_model)

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

        # ---------------- GUI ----------------
        frame = np.ones((400, 700, 3), dtype=np.uint8) * 255
        frame[140:380, 380:700] = cam_frame

        cv2.putText(frame, "SIGN2SOUND - Live Speech Captioning",
                    (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        cv2.putText(frame, "S: Start | E: Stop | Q: Quit | L: Toggle Landmarks",
                    (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

        cv2.putText(frame, current_text,
                    (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)

        cv2.putText(frame, f"Sign: {current_sign}",
                    (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running = False
        elif key == ord("s"):
            listening = True
        elif key == ord("e"):
            listening = False
        elif key == ord("l"):
            DEBUG_LANDMARKS = not DEBUG_LANDMARKS
        elif key == ord("r"):
            RECORDING = True
            current_label = "test_gesture"
            sequence_buffer.clear()
        elif key == ord("t"):
            RECORDING = False

        if listening and not audio_queue.empty():
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    current_text = text

    stream.stop()
    stream.close()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
