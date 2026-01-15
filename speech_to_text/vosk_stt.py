# speech_to_text/vosk_stt.py
# Lightweight Vosk STT module for GUI integration

import threading
import time
import json
import queue
import sys

import sounddevice as sd
from vosk import KaldiRecognizer, Model


class VoskSTT:
    def __init__(
        self,
        model_path="speech_to_text/models/vosk-model-small-en-us-0.15",
        sample_rate=48000,
        device=None,
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.device = device

        self.listening = False
        self.running = False

        self.audio_queue = queue.Queue()
        self.current_text = ""

        # Load model ONCE
        print("Loading Vosk model...")
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

    # ---------------------------
    # Audio callback (UNCHANGED)
    # ---------------------------
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    # ---------------------------
    # Background STT loop
    # ---------------------------
    def _stt_loop(self):
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=8000,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=self.audio_callback,
        ):
            while self.running:
                if not self.listening:
                    time.sleep(0.05)
                    continue

                data = self.audio_queue.get()
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        self.current_text = text

    # ---------------------------
    # Public API (GUI-safe)
    # ---------------------------
    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._stt_loop, daemon=True).start()

    def stop(self):
        self.running = False

    def set_listening(self, value: bool):
        self.listening = value

    def get_text(self):
        return self.current_text
# End of vosk_stt.py