import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from preprocessing.extract_features import load_dataset
from training.label_utils import encode_labels
from models.model import SignLSTM

# -------------------------------
# Config
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_PATH = "checkpoints/final_model.pth"
MODEL_PATH = "checkpoints/final_model_2hand.pth"
os.makedirs("results", exist_ok=True)

# -------------------------------
# Load dataset
# -------------------------------
X, y, labels = load_dataset()
y_encoded, class_to_idx, idx_to_class = encode_labels(y)

# Use same split logic as training (80/20)
num_samples = len(X)
torch.manual_seed(42)
indices = torch.randperm(len(X))

split_idx = int(0.8 * len(X))
test_idx = indices[split_idx:]

X_test = X[test_idx]
y_test = y_encoded[test_idx]


# -------------------------------
# Load model
# -------------------------------
model = SignLSTM(
    input_size=126,
    hidden_size=256,
    num_classes=len(class_to_idx)
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Inference
y_true = []
y_pred = []

with torch.no_grad():
    for i in range(len(X_test)):
        x = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, 126)
        output = model(x)
        pred = torch.argmax(output, dim=1).item()

        y_true.append(y_test[i].item())
        y_pred.append(pred)

# -------------------------------
# Evaluation
acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted"
)
precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
    y_true, y_pred, average=None
)

per_class = []
for idx, label in idx_to_class.items():
    per_class.append({
        "class": label,
        "precision": precision_c[idx],
        "recall": recall_c[idx],
        "f1": f1_c[idx],
        "support": support_c[idx]
    })

df = pd.DataFrame(per_class)
df.to_csv("results/per_class_performance.csv", index=False)
print("Per-class metrics saved.")
# Metrics
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

print("Confusion matrix saved to results/confusion_matrix.png")
