# Preprocessing Pipeline – SIGN2SOUND

This folder contains the preprocessing and feature extraction pipeline used in **SIGN2SOUND** for real-time sign language recognition.

---

## Overview

For Phase 2, preprocessing is designed to be **lightweight, real-time, and reproducible**.

Instead of offline video preprocessing, the system performs **on-the-fly landmark extraction** using **MediaPipe Hands**, followed by sequence construction for LSTM-based classification.

---

## Pipeline Summary

```
Webcam Frame
   ↓
MediaPipe Hands
   ↓
Hand Landmarks (x, y, z)
   ↓
Flattened Landmark Vector (63 features per frame)
   ↓
Temporal Windowing (30 frames)
   ↓
LSTM Input Tensor (30 × 63)
```

---

## Folder Contents

### `extract_features.py`  ⭐ (Primary Component)

* Extracts **hand landmarks** using MediaPipe
* Converts landmarks into **NumPy arrays**
* Handles:

  * Frame normalization
  * Temporal padding/truncation
  * Dataset loading for training
* Used by:

  * `training/train.py`
  * `training/evaluate.py`
  * Real-time inference pipeline

---

### `preprocess.py`

This file is kept to follow the **recommended SIGN2SOUND project structure**.

> For Phase 2, preprocessing is performed directly via `extract_features.py` and real-time MediaPipe extraction.
> No additional offline preprocessing is required.

This file serves as a **placeholder for future pipeline expansion**, such as:

* Multi-dataset integration
* Batch preprocessing
* Exporting preprocessed features

---

### `augmentation.py` (Optional)

Data augmentation is **not used in Phase 2**.

Reason:

* The model is trained on **skeletal landmark representations**, which are already invariant to:

  * Lighting conditions
  * Background noise
  * Camera resolution

Future augmentation ideas (not implemented):

* Temporal jittering
* Noise injection in landmark space
* Sequence warping

---

## Design Rationale

* **Real-time first**: Preprocessing is optimized for live webcam input
* **Minimal latency**: No heavy offline transformations
* **Reproducible**: Same extraction logic used in training and inference
* **Dataset compliant**: Uses MediaPipe-based skeletal keypoints as required by Phase 2

---

## Phase 2 Compliance

✔ Uses **MediaPipe skeletal landmarks**
✔ Consistent preprocessing between training and inference
✔ Matches SIGN2SOUND recommended structure
✔ Fully documented and reproducible

---

## Notes

* All preprocessing logic is intentionally centralized in `extract_features.py`
* The modular structure allows easy extension for Phase 3 or production use

---