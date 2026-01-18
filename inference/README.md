# Inference & Demo – SIGN2SOUND

This module contains the real-time inference and demonstration components
for the SIGN2SOUND system.

---

## Implemented Components

### `infer.py`
- Core inference utilities
- Loads trained model checkpoints
- Performs prediction on landmark sequences
- Used internally by `opencv_gui.py`

### `opencv_gui.py` (PRIMARY DEMO)
- Main real-time interface for SIGN2SOUND
- Captures webcam input
- Extracts hand landmarks using MediaPipe
- Runs live inference using the trained LSTM model
- Applies temporal smoothing to stabilize predictions
- Displays:
  - Live predictions
  - Stable predictions
  - System status
  - Optional speech-to-text output

Run using:
```bash
python -m inference.opencv_gui
