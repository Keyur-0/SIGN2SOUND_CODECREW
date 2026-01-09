# Feature Extraction

This directory contains modules responsible for extracting visual features
from sign language input using **MediaPipe**.

The extracted features are used as inputs to the deep learning model for
**Sign → Text** recognition.

---

## 📌 Purpose
Sign language is inherently visual and temporal. Instead of using raw video
frames, this project extracts **skeletal and landmark-based features**, which:
- Reduce computational complexity
- Improve robustness to lighting/background changes
- Are suitable for time-series modeling (LSTM / Bi-LSTM)

---

## 📂 Files Overview

### `hand_landmarks.py`
- Extracts hand keypoints using MediaPipe Hands
- Captures finger joint positions and hand orientation
- Outputs normalized landmark coordinates

### `pose_estimation.py`
- Extracts upper-body pose landmarks using MediaPipe Pose
- Captures arm, shoulder, and torso movement
- Useful for distinguishing similar hand signs with different arm motion

### `facial_features.py`
- (Optional) Extracts facial landmarks using MediaPipe Face Mesh
- Can be used to capture expressions and mouth movements
- Included for extensibility (not mandatory for all signs)

### `feature_utils.py`
- Shared utility functions for:
  - Normalization
  - Sequence padding/truncation
  - Landmark formatting
- Ensures consistent feature shapes across modules

---

## 🔄 Feature Pipeline
1. Webcam input is captured using OpenCV
2. MediaPipe extracts landmarks frame-by-frame
3. Landmark coordinates are normalized
4. Sequences are aggregated over time
5. Final feature tensors are passed to the model

---

## 📐 Feature Format
- Features are stored as NumPy arrays
- Shape: `(sequence_length, num_features)`
- Compatible with LSTM-based architectures

---

## ⚠️ Notes
- This folder does **not** contain any model training logic
- Raw video data is **not stored**
- Feature extraction runs in real time during inference

---

## 🔮 Extensibility
Additional feature extractors (e.g., depth, optical flow) can be added
without modifying the existing model interface.
