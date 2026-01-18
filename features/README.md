# Feature Extraction – SIGN2SOUND

This module is responsible for converting raw visual input into structured
numerical features suitable for deep learning models.

---

## Core Features Used

### Hand Landmarks (Primary)
- Extracted using **MediaPipe Hands**
- 21 landmarks per hand
- Each landmark contains (x, y, z) coordinates
- Total features per frame: **63**

These hand landmarks form the primary input to the SignLSTM model.

---

## Feature Files Overview

### `hand_landmarks.py`
Handles extraction of hand skeletal landmarks using MediaPipe.
This is the **primary feature extractor** used in training and inference.

### `feature_utils.py`
Utility functions for:
- Normalization
- Flattening landmark arrays
- Sequence shaping

### `pose_estimation.py`
Placeholder for full-body pose extraction.
Not actively used in Phase 2, included for future expansion.

### `facial_features.py`
Placeholder for facial expression features.
Reserved for emotion-aware extensions in later phases.

---

## Feature Shape

- Per frame: `63` values (21 landmarks × 3 coordinates)
- Per sequence: `30 × 63`
- Model input shape: `(batch_size, 30, 63)`

---

## Design Rationale
Hand landmarks were selected as the primary features due to:
- Robustness to lighting and background noise
- Lower computational overhead
- Strong suitability for sign language recognition

---

## Future Extensions
- Multi-hand fusion
- Pose-assisted disambiguation
- Facial expression integration
