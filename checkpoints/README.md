# Model Checkpoints

This directory is used to store trained model weights generated during
the training phase of the SIGN2SOUND project.

---

## 📁 Contents
The following files may be generated after training:

- `best_model.pth`  
  Best-performing model based on validation metrics.

- `final_model.pth`  
  Model saved after the final training epoch.

> ⚠️ These files are **not included** in the repository if they exceed
GitHub’s file size limits.

---

## 📦 Download Instructions
If model checkpoint files exceed 100 MB, they are hosted externally.
Download links will be provided here after training.

Example:
Best Model: <link-to-external-storage>
Final Model: <link-to-external-storage>


---

## 🔁 How to Use a Checkpoint
After downloading a model checkpoint, place it in this directory and run:

```bash
python inference/realtime_demo.py
```
The inference script will automatically load the appropriate model
from this folder.

## 📌 Notes

This directory should only contain model weight files
Do not store raw datasets or logs here
File formats used: .pth (PyTorch)