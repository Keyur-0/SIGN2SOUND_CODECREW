import numpy as np
from pathlib import Path

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 126

DATA_ROOT = Path("data/raw/data")


def load_sample(sample_dir: Path):
    """
    Loads one sample directory (e.g. A/0/)
    Returns: (30, 126) numpy array
    Feature format:
    [LEFT_HAND (63) | RIGHT_HAND (63)]
    """
    frame_files = sorted(
        sample_dir.glob("*.npy"),
        key=lambda x: int(x.stem)
    )

    frames = []
    for f in frame_files:
        frame = np.load(f)

        if frame.shape[0] != FEATURES_PER_FRAME:
            raise ValueError(
                f"Invalid feature size {frame.shape[0]} in {f}, expected {FEATURES_PER_FRAME}"
            )

        frames.append(frame)

    if len(frames) == 0:
        raise ValueError(f"No frames found in {sample_dir}")

    frames = np.array(frames, dtype=np.float32)

    # Pad to 30 frames if needed
    if frames.shape[0] < SEQUENCE_LENGTH:
        pad_count = SEQUENCE_LENGTH - frames.shape[0]
        pad_frames = np.zeros((pad_count, FEATURES_PER_FRAME), dtype=np.float32)
        frames = np.vstack([frames, pad_frames])

    return frames  # (30, 126)


def load_dataset():
    X = []
    y = []
    labels = []

    for label_dir in sorted(DATA_ROOT.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        labels.append(label)

        for sample_dir in sorted(label_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            X.append(load_sample(sample_dir))
            y.append(label)

    return np.array(X), np.array(y), labels


if __name__ == "__main__":
    X, y, labels = load_dataset()
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Labels:", labels[:5])
    print("One sample shape:", X[0].shape)
    