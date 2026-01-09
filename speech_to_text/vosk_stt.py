import json
import queue
import sys

import sounddevice as sd
from vosk import KaldiRecognizer, Model

#------------------------------------------
# Configuration
#------------------------------------------
MODEL_PATH = "speech_to_text/models/vosk-model-small-en-us-0.15"  # Path to Vosk model
SAMPLE_RATE = 48000  # Sample rate for audio recording
MIC_DEVICE_INDEX = 7  # Ryzen HD Audio Controller Stereo Microphone

#Queue to hold audio data
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))

def main():
    print("Loading Vosk model...")
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)

    print("\nSpeech-to-Text Ready")
    print("Press Ctrl+C to stop the program.\n")

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, device=MIC_DEVICE_INDEX, dtype='int16', channels=1, callback=audio_callback):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    print(f"Text: {text}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
               