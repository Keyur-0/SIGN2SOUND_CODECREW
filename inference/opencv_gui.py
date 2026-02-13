import cv2
import numpy as np
import queue
import sys

from speech_to_text.vosk_stt import VoskSTT
from features.hand_landmarks import HandLandmarkExtractor
from inference.infer import load_model, predict_sign
from sign_to_speech.tts import SignToSpeech

# ---------------- CONFIG ----------------
WINDOW_NAME = "SIGN2SOUND"
SEQUENCE_LENGTH = 30

STT_WINDOW_NAME = "Live Captions"
STT_WIDTH = 420
STT_HEIGHT = 140

LOCK_THRESHOLD = 14
MIN_HOLD_FRAMES = 8
COOLDOWN_FRAMES = 15
UNLOCK_THRESHOLD = 7
IDLE_FRAMES = 25

IMPORTANT_WORDS = ["important", "urgent", "understand", "deadline"]

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

candidate_sign = None
candidate_count = 0
cooldown_count = 0
no_hand_count = 0
idle_count = 0

# Sign → Speech
word_buffer = ""
has_appended_current_lock = False

audio_queue = queue.Queue()


# ---------------- HELPERS ----------------
def split_important_words(text: str):
    words = text.split()
    important_indices = []

    for i, w in enumerate(words):
        clean = w.lower().strip(".,!?")
        if clean in IMPORTANT_WORDS:
            important_indices.append(i)

    return words, important_indices

def draw_stt_overlay(text: str):
    frame = np.ones((STT_HEIGHT, STT_WIDTH, 3), dtype=np.uint8) * 250

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    # ---- LABEL ----
    cv2.putText(
        frame,
        "Speech-to-Text",
        (10, 25),
        font,
        0.7,
        (0, 0, 0),
        2
    )

    # underline
    cv2.line(frame, (10, 30), (STT_WIDTH - 10, 30), (0, 0, 0), 1)

    x, y = 10, 55
    space = 8
    line_height = 22

    words = text.split()

    for word in words:
        clean = word.lower().strip(".,!?")

        if clean in IMPORTANT_WORDS:
            label = f"[!] {word}"
            color = (0, 0, 255)
        else:
            label = word
            color = (40, 40, 40)

        (w, h), _ = cv2.getTextSize(label, font, scale, thickness)

        if x + w > STT_WIDTH - 10:
            x = 10
            y += line_height

        cv2.putText(frame, label, (x, y),
                    font, scale, color, thickness)

        x += w + space

    cv2.imshow(STT_WINDOW_NAME, frame)# ---------------- MAIN ----------------
def main():
    global running, listening, current_text
    global live_sign, locked_sign, status_text
    global candidate_sign, candidate_count
    global cooldown_count, no_hand_count, idle_count
    global word_buffer, has_appended_current_lock

    last_stt_text = ""
    tts = SignToSpeech(rate=180)

    # Init STT
    stt = VoskSTT(
        model_path=MODEL_PATH,
        sample_rate=SAMPLE_RATE,
        device=MIC_DEVICE_INDEX
    )
    stt.start()

    extractor = HandLandmarkExtractor()
    model = load_model()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Could not open webcam")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(STT_WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(STT_WINDOW_NAME, STT_WIDTH, STT_HEIGHT)
    cv2.setWindowProperty(
        STT_WINDOW_NAME,
        cv2.WND_PROP_TOPMOST,
        1
    )

    while running:
        ret, cam_frame = cam.read()
        if not ret:
            continue

        if cooldown_count > 0:
            cooldown_count -= 1

        cam_frame = cv2.flip(cam_frame, 1)
        cam_small = cv2.resize(cam_frame, (320, 240))

        # ---------------- FEATURE EXTRACTION ----------------
        features = extractor.extract(cam_small)

        if np.count_nonzero(features) < 20:
            no_hand_count += 1
            idle_count += 1
        else:
            no_hand_count = 0
            idle_count = 0
            sequence_buffer.append(features)

        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)

        # ---------------- HARD RESET ----------------
        if no_hand_count >= UNLOCK_THRESHOLD:
            sequence_buffer.clear()
            live_sign = ""
            locked_sign = ""
            candidate_sign = None
            candidate_count = 0
            cooldown_count = 0
            status_text = "Waiting..."
            has_appended_current_lock = False

        # ---------------- PREDICTION ----------------
        if len(sequence_buffer) < SEQUENCE_LENGTH:
            status_text = "Collecting..."
            live_sign = ""
        else:
            seq_np = np.array(sequence_buffer, dtype=np.float32)
            live_sign = predict_sign(seq_np, model)

            if live_sign == candidate_sign:
                candidate_count += 1
            else:
                candidate_sign = live_sign
                candidate_count = 1

            if (
                candidate_count >= LOCK_THRESHOLD
                and candidate_count >= MIN_HOLD_FRAMES
                and cooldown_count == 0
            ):
                locked_sign = candidate_sign
                status_text = "Sign Stable"
                cooldown_count = COOLDOWN_FRAMES

                if not has_appended_current_lock:
                    word_buffer += locked_sign
                    has_appended_current_lock = True

            elif cooldown_count > 0:
                status_text = "Cooldown..."
            else:
                status_text = "Predicting..."

        # ---------------- AUTO SPEAK ON IDLE ----------------
        if idle_count >= IDLE_FRAMES and word_buffer:
            tts.speak(word_buffer)

            word_buffer = ""
            has_appended_current_lock = False
            candidate_sign = None
            candidate_count = 0
            locked_sign = ""
            status_text = "Waiting..."
            idle_count = 0

        # ---------------- GUI ----------------
        FRAME_W, FRAME_H = 720, 420
        frame = np.ones((FRAME_H, FRAME_W, 3), dtype=np.uint8) * 245

        CAM_W, CAM_H = 280, 210
        CAM_X, CAM_Y = 400, 140
        cam_disp = cv2.resize(cam_small, (CAM_W, CAM_H))
        frame[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cam_disp

        draw_stt_overlay(current_text)

        cv2.rectangle(
            frame,
            (CAM_X - 2, CAM_Y - 2),
            (CAM_X + CAM_W + 2, CAM_Y + CAM_H + 2),
            (0, 0, 0),
            2
        )

        # ---------------- TEXT ----------------
        cv2.putText(frame, "SIGN2SOUND", (40, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)

        cv2.putText(frame, "SIGN:", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        cv2.putText(frame, locked_sign if locked_sign else "-",
                    (150, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(frame, f"Predicting: {live_sign}",
                    (40, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 2)

        # cv2.putText(frame, "Speech:", (40, 170),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        # words, important_idxs = split_important_words(current_text)
        # # x, y = 40, 205

        # # for i, word in enumerate(words):
        # #     color = (0, 0, 255) if i in important_idxs else (30, 30, 30)
        # #     (w, _), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        # #     cv2.putText(frame, word, (x, y),
        # #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # #     x += w + 10

        # x, y = 40, 205
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # scale = 0.7
        # thickness = 2
        # space = 10

        # words = current_text.split()

        # for word in words:
        #     clean = word.lower().strip(".,!?")

        #     # if clean in IMPORTANT_WORDS:
        #     #     label = "[!]important"
        #     #     color = (0, 0, 255)
        #     # else:
        #     #     label = word
        #     #     color = (30, 30, 30)
        #     if clean in IMPORTANT_WORDS:
        #         label = f"[!]{word}"
        #         color = (0, 0, 255)
        #     else:
        #         label = word
        #         color = (30, 30, 30)

        #     (w, _), _ = cv2.getTextSize(label, font, scale, thickness)

        #     cv2.putText(
        #         frame,
        #         label,
        #         (x, y),
        #         font,
        #         scale,
        #         color,
        #         thickness
        #     )

        #     x += w + space

        # if important_idxs:
        #     cv2.putText(frame, "[IMPORTANT]", (40, 235),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, "Status:", (40, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        cv2.putText(frame, status_text, (150, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)

        cv2.putText(frame, f"Word: {word_buffer if word_buffer else '-'}",
                    (40, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

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
        elif key == 32 and word_buffer:
            tts.speak(word_buffer)
            word_buffer = ""
            has_appended_current_lock = False

        if listening:
            new_text = stt.get_text()
            if new_text and new_text != last_stt_text:
                current_text = new_text
                last_stt_text = new_text

    stt.stop()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()