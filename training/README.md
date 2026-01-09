# Model Training

This directory contains scripts and configuration files used for
training and evaluating the **Sign → Text** recognition model in
the SIGN2SOUND project.

The training pipeline is implemented using **PyTorch** and is designed
for time-series classification using skeletal keypoints.

---

## 📌 Training Objective
The goal of training is to learn temporal patterns from sequences of
skeletal landmarks and map them to corresponding sign language labels.

Key objectives:
- Learn spatial–temporal dependencies
- Generalize across different signers
- Enable real-time inference

---

## 📂 Files Overview

### `train.py`
- Main training script
- Loads preprocessed feature sequences
- Trains the LSTM / Bi-LSTM model
- Saves model checkpoints

### `evaluate.py`
- Evaluates trained models on validation/test data
- Computes performance metrics
- Generates confusion matrix and summary reports

### `hyperparams.yaml`
- Centralized configuration for:
  - Learning rate
  - Batch size
  - Number of epochs
  - Model dimensions

### `callbacks.py`
- Optional training utilities such as:
  - Early stopping
  - Model checkpointing
  - Learning rate scheduling

### `README.md`
- Documentation for the training pipeline

---

## 🔄 Training Pipeline
1. Load extracted skeletal features
2. Split data into training and validation sets
3. Initialize model and optimizer
4. Train model over multiple epochs
5. Evaluate performance on validation data
6. Save best and final model checkpoints

---

## 📈 Evaluation Metrics
The following metrics are used to assess model performance:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Evaluation outputs are stored in:
```bash
results/
```

---

## ▶️ How to Train
```bash
python training/train.py
```

---

## ▶️ How to Evaluate
```bash
python training/evaluate.py
```

---

## ⚠️ Notes

- Training is performed on CPU by default
- GPU acceleration is optional and not required
- Raw datasets are not stored in this repository
- Model checkpoints are saved in checkpoints/

---

## 🔮 Future Improvements

- Hyperparameter tuning
- Data augmentation strategies
- Transformer-based sequence models
- Cross-dataset evaluation


---
