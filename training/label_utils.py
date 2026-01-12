import numpy as np


def encode_labels(y):
    classes = sorted(set(y))

    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    y_encoded = np.array([class_to_idx[label] for label in y], dtype=np.int64)

    return y_encoded, class_to_idx, idx_to_class
