#TODO: Test if the OpenCV GUI integration with Vosk STT works as intended
import mediapipe as mp

import cv2
import numpy as np

import threading
import time
import json
import queue
import sys

import sounddevice as sd
from vosk import KaldiRecognizer, Model

# DEBUG VARIABLES
DEBUG_LANDMARKS = False

#MEDIAPIPE VARIABLES FOR LSTMS
SEQUENCE_LENGTH = 30   # frames per sequence
sequence_buffer = []  # rolling buffer of landmark vectors

#vosk STT VARIABLES
listening = False
running = True
MODEL_PATH = "speech_to_text/models/vosk-model-small-en-us-0.15"  # Path to Vosk model
SAMPLE_RATE = 48000  # Sample rate for audio recording
MIC_DEVICE_INDEX = None  # use default microphone

#OPEN CV GUI VARIABLES
WINDOW_NAME = "SIGN2SOUND"
current_text = ""

#Queue to hold audio data
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))

def main():
    global running, current_text, listening, DEBUG_LANDMARKS
    print("Loading Vosk model...")
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    #Initialize OpenCV Video Capture
    print("Opening webcam...")
    cam = cv2.VideoCapture(0)
    
    #initialize mediapipe hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    #Start audio stream
    print("Starting audio stream...")
    stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, device=MIC_DEVICE_INDEX, dtype='int16', channels=1, callback=audio_callback)
    stream.start()
    
    #Initialize OpenCV window
    print("OpenCV GUI started")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while running:
        #Read frame from webcam
        ref, cam_frame = cam.read()
        if not ref:
            continue

        cam_frame = cv2.resize(cam_frame, (320, 240))
        cam_frame = cv2.flip(cam_frame, 1)

        #Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # ---- Landmark extraction (21 x 3 = 63) ----
                landmark_vector = []
                for lm in hand_landmarks.landmark:
                    landmark_vector.extend([lm.x, lm.y, lm.z])

                landmark_vector = np.array(landmark_vector, dtype=np.float32)
                # landmark_vector.shape == (63,)

                # ---- Sequence windowing ----
                sequence_buffer.append(landmark_vector)

                # Keep only last SEQUENCE_LENGTH frames
                if len(sequence_buffer) > SEQUENCE_LENGTH:
                    sequence_buffer.pop(0)

                # When buffer is full, we have a valid sequence
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    sequence = np.stack(sequence_buffer)  # shape: (30, 63)

                    if DEBUG_LANDMARKS:
                        print("Sequence shape:", sequence.shape)


                if DEBUG_LANDMARKS:
                    print("Landmark vector shape:", landmark_vector.shape)
                    print(landmark_vector)

                # ---- Visualization (unchanged) ----
                mp_drawing.draw_landmarks(
                    cam_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )


        #Create white frame
        frame = np.ones((400, 700, 3), dtype=np.uint8) * 255

        frame[140:380, 380:700] = cam_frame
        cv2.putText(frame, "Webcam (Sign Input)", (360, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 0),1)

        #Title
        cv2.putText(frame, "SIGN2SOUND - Live Speech Captioning", (40, 40), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2)

        #instructions
        cv2.putText(frame, "S: Start | E: Stop | Q: Quit | L: Toggle Landmarks", (40,80), cv2.FONT_HERSHEY_SIMPLEX,0.6,(50,50,50),1)

        #Display recognized text
        cv2.putText(frame, current_text, (40, 150),cv2.FONT_HERSHEY_SIMPLEX,0.9,(20, 20, 20),2)

        
        cv2.imshow(WINDOW_NAME, frame)

        #Open CV key controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            running = False
        elif key == ord('s'):
            listening = True
        elif key == ord('e'):
            listening = False
        elif key == ord('l'):
            DEBUG_LANDMARKS = not DEBUG_LANDMARKS
            print(f"[DEBUG] Landmark logging: {DEBUG_LANDMARKS}")

            
        #Speech to text processing
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