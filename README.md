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
- Live speech captions are rendered in a real-time OpenCV GUI with start/stop controls.

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
- Deep learning model (PyTorch LSTM-based sequence model)
- Converts signs into text output

### 🔵 Sign → Speech (Scalable / Optional)
- Converts recognized sign text into audio
- Implemented as a **modular extension**
- Can be enabled without retraining the model

---

## 🧠 System Architecture
```

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
```
SIGN2SOUND_CodeCrew/
│
├── checkpoints/                     # Trained model weights
│   ├── best_model.pth
│   ├── final_model.pth
│   └── README.md
│
├── data/                            # Dataset documentation (IEEE compliant)
│   ├── README.md
│   └── statistics.txt
│
├── preprocessing/                   # Data preprocessing pipeline
│   ├── preprocess.py
│   ├── extract_features.py
│   ├── augmentation.py
│   └── README.md
│
├── features/                        # Feature extraction modules
│   ├── hand_landmarks.py
│   ├── pose_estimation.py
│   ├── facial_features.py
│   ├── feature_utils.py
│   └── README.md
│
├── models/                          # Model architecture definitions
│   ├── model.py
│   ├── custom_layers.py
│   ├── loss.py
│   └── README.md
│
├── training/                        # Training & evaluation pipeline
│   ├── train.py
│   ├── evaluate.py
│   ├── dataset.py
│   ├── label_utils.py
│   ├── callbacks.py
│   ├── hyperparams.yaml
│   └── README.md
│
├── inference/                       # Real-time inference & demo
│   ├── infer.py
│   ├── opencv_gui.py                # Main Phase-2 demo
│   ├── realtime_demo.py
│   ├── tts.py
│   ├── utils.py
│   └── README.md
│
├── speech_to_text/                  # Offline Speech → Text module
│   ├── vosk_stt.py
│   ├── models/                      # Vosk language models
│   └── __pycache__/
│
├── notebooks/                       # Experiments & analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiment.ipynb
│   ├── 03_results_visualization.ipynb
│   └── README.md
│
├── results/                         # Evaluation outputs
│   ├── accuracy_curves.png
│   ├── loss_curves.png
│   ├── confusion_matrix.png
│   ├── metrics.json
│   ├── per_class_performance.csv
│   ├── training_log.txt
│   ├── plot_curves.py
│   └── sample_outputs/
│       ├── sample_1.png
│       ├── sample_2.png
│       └── predictions.txt
│
├── docs/                            # Documentation & reports
│   ├── architecture_diagram.png
│   ├── system_pipeline.png
│   ├── dataset_preprocessing.md
│   ├── training_details.md
│   └── technical_report.pdf
│
├── tests/                           # Unit tests
│   ├── test_inference.py
│   └── test_model.py
│
├── README.md                        # Main project documentation
├── requirements.txt                # Dependencies
├── LICENSE
└── .gitignore
```
---

## ▶️ How to Run the Project

### 1️⃣ Environment Setup
```bash
python -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Run Speech → Text (Vosk)
```bash
python speech_to_text/vosk_stt.py
```

### 3️⃣ Run Sign → Text (Webcam Demo)
```bash
python -m inference.opencv_gui
```

## 🎥 Demo

A real-time multimodal demo is provided showcasing both **Sign → Text** and **Speech → Text** capabilities of the SIGN2SOUND system through a unified OpenCV-based interface.

### The demo demonstrates:

* **Live webcam capture**
* **Hand landmark extraction** using MediaPipe
* **Real-time Sign → Text recognition** using an LSTM-based model
* **Temporal prediction smoothing** to stabilize sign outputs
* **Offline Speech → Text transcription** using Vosk
* **Live visual feedback** including:

  * Current sign prediction
  * Stable sign output
  * Speech transcription
  * System status indicators

The Sign → Text and Speech → Text pipelines operate independently but are visualized together to demonstrate **bidirectional accessibility**.

### Running the demo:

```bash
python -m inference.opencv_gui
```

### Sample Outputs:

Screenshots and example predictions from the demo are available in:

```bash
results/sample_outputs/
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
## 🧪 Testing

Basic unit tests are provided to verify model forward passes and inference
utilities. These tests are lightweight and designed to ensure functional
correctness without requiring external hardware or datasets.

Run tests using:
```bash
python -m tests.test_model
python -m tests.test_inference
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
This project is released under the MIT License.

---

## 🙏 Acknowledgements

IEEE DataPort for providing datasets
MediaPipe for skeletal landmark extraction
Vosk for offline speech recognition

---
