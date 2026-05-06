"""
Image dataset loader for gesture and fatigue classification.
Reads from data/CompVision_DataSet/{split}/images/{task}/{class}/
"""

import os
import cv2
import numpy as np
from pathlib import Path

# ── Label maps ────────────────────────────────────────────────────────────────
GESTURE_CLASSES  = ["open_palm", "thumbs_up", "wrong_gesture"]
FATIGUE_CLASSES  = ["alert", "eyes_closed", "head_pose",
                    "tired_expression", "yawn"]

GESTURE_LABEL  = {c: i for i, c in enumerate(GESTURE_CLASSES)}
FATIGUE_LABEL  = {
    "alert":            0,
    "eyes_closed":      1,
    "head_down":        2,
    "head_tilt_left":   2,
    "head_tilt_right":  2,
    "tired_expression": 3,
    "yawn":             4,
}

IMG_SIZE = (224, 224)   # MobileNetV2 input size


# ── Augmentation helpers ──────────────────────────────────────────────────────
def _augment(img: np.ndarray) -> list[np.ndarray]:
    """Return a list of augmented copies of img (used only for training)."""
    augmented = []

    # Horizontal flip
    augmented.append(cv2.flip(img, 1))

    # Brightness shift  ±30
    for delta in (-30, 30):
        shifted = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        augmented.append(shifted)

    # Small rotation ±10°
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    for angle in (-10, 10):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        augmented.append(cv2.warpAffine(img, M, (w, h)))

    # Gaussian blur (simulates motion)
    augmented.append(cv2.GaussianBlur(img, (5, 5), 0))

    return augmented


# ── Core loader ───────────────────────────────────────────────────────────────
def load_image_dataset(
    dataset_root: str,
    task: str,          # "gesture" or "fatigue"
    split: str,         # "train", "val", or "test"
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images for a given task/split.

    Returns
    -------
    X : np.ndarray  shape (N, 224, 224, 3), dtype float32, values 0-1
    y : np.ndarray  shape (N,),             dtype int64
    """
    label_map = GESTURE_LABEL if task == "gesture" else FATIGUE_LABEL
    img_dir = Path(dataset_root) / split / "images" / task

    images, labels = [], []

    for class_name, label_idx in label_map.items():
        class_dir = img_dir / class_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(label_idx)

            if augment:
                for aug in _augment(img):
                    images.append(aug)
                    labels.append(label_idx)

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int64)
    return X, y


def get_class_names(task: str) -> list[str]:
    return GESTURE_CLASSES if task == "gesture" else FATIGUE_CLASSES


def dataset_summary(dataset_root: str) -> None:
    """Print per-class counts across all splits."""
    for split in ("train", "val", "test"):
        for task in ("gesture", "fatigue"):
            X, y = load_image_dataset(dataset_root, task, split, augment=False)
            names = get_class_names(task)
            print(f"\n[{split.upper()}] {task}")
            for i, name in enumerate(names):
                count = int((y == i).sum())
                print(f"  {name:20s} {count}")
            print(f"  {'TOTAL':20s} {len(y)}")


if __name__ == "__main__":
    root = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "CompVision_DataSet"
    )
    dataset_summary(root)
