# SIGN2SOUND – CodeCrew

## 🔊 Bridging Communication Through AI

SIGN2SOUND is a multimodal accessibility system designed to reduce communication barriers between **speech users and sign language users**.  
The project focuses on **real-time Speech-to-Text and Sign-to-Text translation**, with a scalable architecture that supports **Sign-to-Speech** as a future extension.

This repository contains the **Phase-2 implementation** for the SIGN2SOUND challenge.

---

## 🎯 Problem Statement
Communication between hearing individuals and sign language users remains a major accessibility challenge. Existing solutions are often expensive, language-specific, or not real-time.

SIGN2SOUND aims to:
- Enable **real-time interaction**
- Use **AI-based recognition**
- Remain **lightweight and scalable**
- Support **bidirectional communication**

---

## ✅ Core Features

### 🟢 Speech → Text (Implemented)
- Real-time speech recognition using **Vosk**
- Offline processing (no internet required)
- Converts spoken language into readable text

#### Audio Input Note
By default, the Speech → Text module captures audio from a microphone device.
For online meeting transcription (e.g., Zoom, Google Meet), the system can
capture speaker audio using system-level loopback or monitor devices
(PipeWire / PulseAudio).

This enables transcription of meeting audio without modifying the core
speech recognition pipeline.

### 🟢 Sign → Text (Implemented)
- Real-time sign language recognition using **skeletal keypoints**
- Uses **MediaPipe** for landmark extraction
- Deep learning model (PyTorch LSTM/Bi-LSTM)
- Converts signs into text output

### 🔵 Sign → Speech (Scalable / Optional)
- Converts recognized sign text into audio
- Implemented as a **modular extension**
- Can be enabled without retraining the model

---

## 🧠 System Architecture
```bash

                    ┌────────────┐
Speech User ──Mic──▶│   VOSK     │──▶ Text Output
                    └────────────┘

Sign User ──Camera──▶ MediaPipe ─▶ LSTM Model ─▶ Text Output
                                      │
                                      ▼
                               (Optional)
                                  TTS
```
---

## 📊 Dataset Information
This project uses an **IEEE DataPort dataset** as required by the challenge:

- **Indian Sign Language Skeletal-point NumPy Array (MediaPipe)**
  - Contains pre-extracted skeletal keypoints
  - Suitable for time-series modeling
  - Enables fast training and real-time inference

📌 Due to dataset licensing, **raw data is not included** in this repository.  
### Dataset sources and usage instructions are documented in:
```bash
 data/README.md
```
---

## 🏗️ Project Structure (Simplified)
```bash
SIGN2SOUND_CodeCrew/
├── data/ # Dataset documentation (IEEE compliant)
├── preprocessing/ # Data preprocessing pipeline
├── features/ # MediaPipe landmark extraction
├── models/ # PyTorch model architecture
├── training/ # Training & evaluation scripts
├── inference/ # Real-time inference & demo
├── results/ # Metrics, graphs, outputs
├── checkpoints/ # Trained model weights
├── docs/ # Diagrams & technical report
├── README.md
└── requirements.txt
```
---

## ▶️ How to Run the Project

### 1️⃣ Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Run Speech → Text (Vosk)
```bash
python speech_to_text/vosk_stt.py
```

### 3️⃣ Run Sign → Text (Webcam Demo)
```bash
python inference/realtime_demo.py
```

---

## 📈 Evaluation & Results
### The system is evaluated using standard classification metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion Matrix
    - Training & Validation loss curves

### All evaluation outputs are stored in:
```bash
    results/
```

---

## 📄 Documentation

### Comprehensive documentation is available in the docs/ directory:
- architecture_diagram.png – Model architecture overview
- system_pipeline.png – End-to-end system flow
- technical_report.pdf – Detailed technical explanation
- dataset_preprocessing.md – Data preparation details
- training_details.md – Training configuration & procedure

---

## 🔮 Future Enhancements

- Full Sign → Speech integration
- Sentence-level and continuous sign recognition
- Support for additional sign languages
- Bidirectional conversational interface
- Deployment on web and mobile platforms

---

## 👥 Team
 CodeCrew
 SIGN2SOUND Challenge – Phase 2

---

## 📜 License
This project is released under the MIT License (or applicable license).

---

## 🙏 Acknowledgements

IEEE DataPort for providing datasets
MediaPipe for skeletal landmark extraction
Vosk for offline speech recognition

---