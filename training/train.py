import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing.extract_features import load_dataset
from training.dataset import SignDataset
from training.label_utils import encode_labels
from models.model import SignLSTM
from training.callbacks import ModelCheckpoint

import json
import os

# 1. Configuration
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Metrics storage
train_losses = []
val_losses = []
train_accs = []
val_accs = []
# Train log file
os.makedirs("results", exist_ok=True)
log_file = open("results/training_log.txt", "w")

print("Using device:", DEVICE)


# 2. Load dataset
# Load dataset
X, y, labels = load_dataset()

# Encode labels
y_encoded, class_to_idx, idx_to_class = encode_labels(y)
num_classes = len(class_to_idx)

print("Total samples:", X.shape[0])


# 3. Encode labels
y_encoded, class_to_idx, idx_to_class = encode_labels(y)
num_classes = len(class_to_idx)


# 4. Train / Validation split
num_samples = len(X)
indices = torch.randperm(num_samples)

split_idx = int(0.8 * num_samples)
train_idx = indices[:split_idx]
val_idx = indices[split_idx:]

X_train, y_train = X[train_idx], y_encoded[train_idx]
X_val, y_val = X[val_idx], y_encoded[val_idx]

print("Train samples:", len(X_train))
print("Validation samples:", len(X_val))


# 5. PyTorch datasets & loaders
train_dataset = SignDataset(X_train, y_train)
val_dataset = SignDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# 6. Model, loss, optimizer
model = SignLSTM(
    input_size=63,
    hidden_size=256,
    num_classes=num_classes
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
checkpoint = ModelCheckpoint("checkpoints/best_model.pth")


# 7. Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    train_acc = correct / total

    # ---------------------------
    # Validation
    # ---------------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()

    val_acc = val_correct / val_total
    checkpoint.step(model, val_acc, epoch + 1)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}"
    )
    log_line = (
    f"Epoch [{epoch+1}/{EPOCHS}] | "
    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}"
    )

    print(log_line)
    log_file.write(log_line + "\n")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)


# 8. Save model

torch.save(model.state_dict(), "checkpoints/final_model.pth")
print("Model saved to checkpoints/sign_lstm.pth")


os.makedirs("results", exist_ok=True)

metrics = {
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_accuracy": train_accs,
    "val_accuracy": val_accs
}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

log_file.close()
print("Training metrics saved to results/metrics.json")