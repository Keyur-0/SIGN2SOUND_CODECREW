import torch
import os

class ModelCheckpoint:
    """
    Saves the model when validation accuracy improves.
    """

    def __init__(self, save_path="checkpoints/sign_lstm.pth"):
        self.best_acc = 0.0
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def step(self, model, val_acc, epoch):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            torch.save(model.state_dict(), self.save_path)
            print(
                f"[Checkpoint] Epoch {epoch}: "
                f"Val Acc improved to {val_acc:.3f}. Model saved."
            )
