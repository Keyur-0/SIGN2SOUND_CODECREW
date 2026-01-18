# Dataset Information – SIGN2SOUND

## Dataset Used
- **Indian Sign Language Skeletal-point NumPy Array**
- Landmark extraction via **MediaPipe**
- Source: **IEEE DataPort** (official SIGN2SOUND resource)

## Dataset Description
The dataset contains pre-extracted skeletal hand landmarks generated using
MediaPipe. Each sample represents a sign gesture as a **temporal sequence**
of hand keypoints, making it suitable for sequence-based deep learning models
such as LSTMs.

## Dataset Format
- Each frame contains **63 values**  
  (21 hand landmarks × x, y, z coordinates)
- Each sample is represented as a **fixed-length sequence of 30 frames**
- Labels correspond to **alphabet-level Indian Sign Language classes**

## Usage in This Project
- Used for training and evaluating the **Sign → Text recognition model**
- Input format: NumPy arrays of landmark sequences
- Sequences are dynamically loaded during training
- Train / validation split: **80% / 20%**

## Dataset Statistics
- Number of samples: **3000**
- Number of classes: **25** (A–Z, excluding R)
- Sequence length: **30 frames**
- Feature dimension: **63 skeletal landmarks per frame**
- Train / validation split: **80% / 20%**

## Licensing Notice
Due to IEEE DataPort licensing restrictions, **raw dataset files are not included**
in this repository.

## How to Obtain the Dataset
1. Visit the IEEE DataPort link provided in the SIGN2SOUND challenge resources
2. Download the *Indian Sign Language Skeletal-point NumPy Array* dataset
3. Follow the preprocessing steps described in:
   `preprocessing/README.md`

Note: The `data/processed/` directory is intentionally left empty to avoid
redistribution of IEEE DataPort licensed datasets. All preprocessing is
performed dynamically via `preprocessing/extract_features.py`.