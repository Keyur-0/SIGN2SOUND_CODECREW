import json
import matplotlib.pyplot as plt
import os

with open("results/metrics.json") as f:
    metrics = json.load(f)

epochs = range(1, len(metrics["train_loss"]) + 1)

os.makedirs("results", exist_ok=True)

# ---- LOSS CURVE ----
plt.figure()
plt.plot(epochs, metrics["train_loss"], label="Train Loss")
plt.plot(epochs, metrics["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("results/loss_curves.png")
plt.close()

# ---- ACCURACY CURVE ----
plt.figure()
plt.plot(epochs, metrics["train_accuracy"], label="Train Accuracy")
plt.plot(epochs, metrics["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("results/accuracy_curves.png")
plt.close()

print("Saved loss_curves.png and accuracy_curves.png")
