#TODO: Test if the OpenCV GUI integration with Vosk STT works as intended
import cv2
import numpy as np

import threading
import time
import json
import queue
import sys

import sounddevice as sd
from vosk import KaldiRecognizer, Model
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
    global running, current_text, listening
    print("Loading Vosk model...")
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    
    print("Starting audio stream...")

    stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, device=MIC_DEVICE_INDEX, dtype='int16', channels=1, callback=audio_callback)
    stream.start()
    
    print("OpenCV GUI started")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while running:
        frame = np.ones((400, 700, 3), dtype=np.uint8) * 255

        #Title
        cv2.putText(frame, "SIGN2SOUND - Live Speech Captioning", (80,200), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2)

        #instructions
        cv2.putText(frame, "S: Start | E: Stop | Q: Quit", (220,230), cv2.FONT_HERSHEY_SIMPLEX,0.6,(50,50,50),1)

        #Display recognized text
        cv2.putText(frame, current_text, (40, 180),cv2.FONT_HERSHEY_SIMPLEX,0.9,(20, 20, 20),2)

        
        cv2.imshow(WINDOW_NAME, frame)

        #Open CV key controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            running = False
        elif key == ord('s'):
            listening = True
        elif key == ord('e'):
            listening = False
            
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
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()