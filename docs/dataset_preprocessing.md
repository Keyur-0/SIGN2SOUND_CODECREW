# Dataset Preprocessing – SIGN2SOUND

## Overview

This document describes the preprocessing pipeline used for the SIGN2SOUND
Sign Language Recognition system. The goal of preprocessing is to transform
raw skeletal landmark data into fixed-length, model-ready sequences suitable
for sequence-based deep learning models.

---

## Input Dataset

* **Dataset**: Indian Sign Language Skeletal-point Dataset
* **Source**: IEEE DataPort (official SIGN2SOUND resource)
* **Landmark Extraction**: MediaPipe
* **Data Format**: NumPy arrays

Each sample consists of hand skeletal landmarks extracted per video frame.

---

## Landmark Representation

* **Landmarks per frame**: 21 hand landmarks
* **Coordinates per landmark**: (x, y, z)
* **Total features per frame**:
  `21 × 3 = 63`

Each frame is represented as a **63-dimensional feature vector**.

---

## Temporal Sequencing

Sign gestures are temporal in nature. To model this:

* Each gesture is represented as a **fixed-length sequence**
* **Sequence length**: 30 frames
* Sequences shorter than 30 frames are **discarded**
* Longer sequences are **truncated** to maintain uniform length

Final input shape per sample:

```
(30 frames × 63 features)
```

---

## Label Encoding

* Each gesture is assigned a **class label**
* Alphabet-level sign classes are used
* A total of **25 classes** are included
  (A–Z, excluding ambiguous signs)

Labels are encoded into integer indices during training using a consistent
mapping to ensure reproducibility across training and evaluation.

---

## Dataset Splitting

The dataset is split dynamically during loading:

* **Training set**: 80%
* **Validation/Test set**: 20%
* The split is randomized using a **fixed random seed** to ensure reproducibility

No overlap exists between training and evaluation samples.

---

## Data Augmentation

A placeholder module (`augmentation.py`) is included to support future
augmentation strategies such as:

* Temporal jittering
* Noise injection
* Spatial perturbations

**Note**: No data augmentation is applied in the current Phase 2 submission.

---

## Preprocessing Output

After preprocessing, the dataset is loaded into memory with the following
structure:

* Feature tensor shape: `(N, 30, 63)`
* Label tensor shape: `(N,)`

Where `N` is the total number of valid gesture sequences.

---

## Reproducibility

* Fixed sequence length
* Fixed feature dimensionality
* Deterministic train/validation split
* Consistent label encoding

These steps ensure consistent and reproducible results across experiments.

---

## Licensing Note

Due to IEEE DataPort licensing restrictions, **raw dataset files are not included**
in this repository. Users must download the dataset separately and follow the
preprocessing steps described here.

---

## Related Files

* `preprocessing/extract_features.py` – Dataset loading and sequence preparation
* `preprocessing/preprocess.py` – Pipeline entry point
* `data/statistics.txt` – Dataset statistics summary