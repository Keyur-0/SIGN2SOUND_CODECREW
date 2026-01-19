# Training Details – SIGN2SOUND

## Overview

This document describes the training procedure, model configuration, and
evaluation strategy used for the SIGN2SOUND Sign Language Recognition system.
The model is designed to recognize alphabet-level sign gestures from temporal
hand landmark sequences.

---

## Model Architecture

The Sign → Text model is based on a **Long Short-Term Memory (LSTM)** network
suited for sequential data.

### Architecture Summary

* **Input**: Temporal sequence of hand landmarks
  Shape: `(30 frames × 63 features)`
* **LSTM Layers**:

  * Number of layers: 2
  * Hidden size: 256
  * Dropout: 0.3
  * Batch-first configuration
* **Fully Connected Layer**:

  * Input features: 256
  * Output features: 25 (sign classes)
* **Output**:

  * Raw logits
  * Softmax applied implicitly via loss function during training

---

## Loss Function

* **Criterion**: CrossEntropyLoss
* Suitable for multi-class classification
* Operates directly on model logits and integer-encoded labels

---

## Optimization Strategy

* **Optimizer**: Adam
* **Learning rate**: Configured via `hyperparams.yaml`
* Adam was selected for its stability and fast convergence on sequence models

---

## Training Configuration

* **Batch size**: Defined in `hyperparams.yaml`
* **Epochs**: ~40 (early stopping not applied)
* **Device**:

  * GPU if available
  * CPU fallback supported

Training is performed using PyTorch with automatic differentiation.

---

## Dataset Split

* **Training set**: 80%
* **Validation set**: 20%
* Split is randomized with a fixed seed for reproducibility

---

## Training Process

1. Load preprocessed landmark sequences
2. Encode labels into integer indices
3. Initialize the LSTM model
4. Forward pass through the model
5. Compute loss using CrossEntropyLoss
6. Backpropagate gradients
7. Update weights using Adam optimizer
8. Repeat for each epoch

Training and validation metrics are logged per epoch.

---

## Evaluation Strategy

Model evaluation is performed using a held-out validation set.

### Metrics Used

* Accuracy
* Precision (weighted)
* Recall (weighted)
* F1-score (weighted)
* Confusion Matrix

Evaluation is implemented in:

```
training/evaluate.py
```

---

## Performance Summary

* **Validation Accuracy**: ~74–77%
* **F1-score**: ~0.72–0.75
* Performance varies slightly due to randomized dataset splits

Confusion matrix and metric visualizations are stored in the `results/` folder.

---

## Model Checkpoints

Two checkpoints are saved:

* `best_model.pth` – Best performing model during training
* `final_model.pth` – Final epoch model

Checkpoints are stored in:

```
checkpoints/
```

---

## Reproducibility

* Fixed random seeds
* Consistent preprocessing pipeline
* Deterministic train/validation split
* Explicit model hyperparameters

---

## Limitations

* No data augmentation applied
* Limited dataset size
* Alphabet-only gesture set
* Single-hand landmark focus

These limitations will be addressed in future phases.

---

## Related Files

* `training/train.py` – Training loop implementation
* `training/evaluate.py` – Evaluation script
* `training/hyperparams.yaml` – Hyperparameter configuration
* `results/` – Metrics and visualizations