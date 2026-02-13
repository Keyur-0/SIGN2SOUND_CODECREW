# sign_to_speech/tts.py
import pyttsx3

class SignToSpeech:
    def __init__(self, rate=150):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)

    def speak(self, text: str):
        if text:
            self.engine.say(text)
            self.engine.runAndWait()