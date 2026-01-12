import numpy as np

from preprocessing.extract_features import load_dataset
from training.label_utils import encode_labels
from training.dataset import SignDataset
# Load dataset
X, y = load_dataset()

# Encode labels
y_encoded, class_to_idx, idx_to_class = encode_labels(y)

# Shuffle indices
num_samples = len(X)
indices = np.arange(num_samples)
np.random.seed(42)
np.random.shuffle(indices)

# Split index
split_idx = int(0.8 * num_samples)

train_idx = indices[:split_idx]
val_idx = indices[split_idx:]

# Create splits
X_train = X[train_idx]
y_train = y_encoded[train_idx]

X_val = X[val_idx]
y_val = y_encoded[val_idx]

print("Train samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])

# PyTorch datasets
train_dataset = SignDataset(X_train, y_train)
val_dataset = SignDataset(X_val, y_val)
