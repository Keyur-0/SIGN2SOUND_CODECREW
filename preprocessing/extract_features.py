import numpy as np
from pathlib import Path

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 63

DATA_ROOT = Path("data/raw/data")


def load_sample(sample_dir: Path):
    """
    Loads one sample directory (e.g. A/0/)
    Returns: (30, 63) numpy array
    """
    frame_files = sorted(sample_dir.glob("*.npy"))

    frames = []
    for f in frame_files:
        frame = np.load(f)          # (126,)
        frame = frame[:63]           # take one hand
        frames.append(frame)

    frames = np.array(frames)        # (29, 63)

    # Pad last frame if needed
    if frames.shape[0] < SEQUENCE_LENGTH:
        pad_count = SEQUENCE_LENGTH - frames.shape[0]
        pad_frames = np.repeat(frames[-1][None, :], pad_count, axis=0)
        frames = np.vstack([frames, pad_frames])

    return frames  # (30, 63)


def load_dataset():
    X = []
    y = []

    for label_dir in sorted(DATA_ROOT.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name  # A, B, C, ...

        for sample_dir in label_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            sequence = load_sample(sample_dir)
            X.append(sequence)
            y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_dataset()
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Example label:", y[0])
    print("Example sequence:", X[0])
