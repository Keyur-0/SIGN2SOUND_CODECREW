# speech_to_text/system_audio_stt_test.py
# Phase 3 prototype: System Audio (PipeWire Loopback) → VOSK STT

import json
import queue
import sys
import time

import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ---------------- CONFIG ----------------
# ---------------- CONFIG ----------------
MODEL_PATH = "speech_to_text/models/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 48000

SYSTEM_AUDIO_DEVICE = None # ✅ pipewire

# 🔴 CHANGE THIS to your system monitor source
# SYSTEM_AUDIO_DEVICE = "alsa_output.pci-0000_03_00.6.HiFi__Speaker__sink.monitor"

# ---------------- SETUP ----------------
print("Loading Vosk model...")
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))

print("Starting system audio STT...")
print("Capturing system sound (meet / media audio)...")
print("Press Ctrl+C to stop.\n")

# ---------------- MAIN LOOP ----------------
with sd.RawInputStream(
    samplerate=48000,
    blocksize=8000,
    dtype="int16",   # ✅ important
    channels=1,      # ✅ important
    device=None,     # ✅ default input
    callback=audio_callback,
):
    try:
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    print("🗣️ System Audio:", text)
    except KeyboardInterrupt:
        print("\nStopping system audio STT.")