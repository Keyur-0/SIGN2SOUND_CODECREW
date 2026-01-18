# Models – SIGN2SOUND

## Overview

This directory contains the **deep learning architecture** used for
Sign Language recognition in the SIGN2SOUND system.

The model is designed to process **temporal sequences of skeletal hand landmarks**
and predict **alphabet-level sign classes**.

---

## Model Architecture

### Primary Model: `SignLSTM` (`model.py`)

The system uses a **Long Short-Term Memory (LSTM)** network to model
the temporal dynamics of sign language gestures.

**Why LSTM?**

* Sign language is inherently **temporal**
* Hand poses change over time, not frame-by-frame
* LSTMs capture motion patterns across frames
* Lightweight and suitable for real-time inference

---

## Input Representation

* **Input shape:** `(30, 63)`

  * 30 frames per gesture sequence
  * 63 features per frame
    (21 hand landmarks × x, y, z coordinates)
* Input is normalized skeletal landmark data extracted via **MediaPipe**

---

## Output Representation

* **Output:** Softmax probabilities over sign classes
* **Number of classes:** 25 alphabet-level signs
  (A–Z, excluding motion-based letters)
* Final prediction obtained using `argmax`

---

## Model Components

### `model.py`

* Defines the `SignLSTM` architecture
* Layers include:

  * LSTM layer (hidden size = 256)
  * Fully connected classification layer
* Outputs class logits for each sequence

---

### `custom_layers.py`

* Reserved for future architectural extensions
* Examples:

  * Attention mechanisms
  * Temporal pooling layers
* Currently not required for Phase 2

---

### `loss.py`

* Reserved for custom loss functions
* Current training uses standard `CrossEntropyLoss`
* File retained for extensibility

---

## Training & Checkpoints

* Model is trained using the pipeline in `training/train.py`
* Best-performing model (based on validation accuracy) is saved as:

  ```
  checkpoints/best_model.pth
  ```
* Final trained model is saved as:

  ```
  checkpoints/final_model.pth
  ```

---

## Design Considerations

* Optimized for **real-time inference**
* Low latency for webcam-based demos
* Modular design allows:

  * Model upgrades
  * Additional modalities (pose, face)
  * Multi-hand extensions

---

## Future Improvements

* Attention-based temporal modeling
* Transformer-based sequence models
* Multi-modal fusion (hand + pose + facial expressions)

---

## Summary

The SIGN2SOUND model architecture is:

* **Temporal**
* **Lightweight**
* **Real-time capable**
* **Designed for ethical, skeletal-only data**

This makes it suitable for both research evaluation and practical deployment.