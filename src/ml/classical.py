"""
Classical ML pipeline: HOG features + SVM / Random Forest classifiers.
Trains and persists models using joblib.
"""

import os
import pickle
import numpy as np
import cv2
from pathlib import Path

# ── HOG parameters ────────────────────────────────────────────────────────────
HOG_WIN_SIZE    = (64, 64)
HOG_BLOCK_SIZE  = (16, 16)
HOG_BLOCK_STRIDE= (8, 8)
HOG_CELL_SIZE   = (8, 8)
HOG_NBINS       = 9

_hog = cv2.HOGDescriptor(
    HOG_WIN_SIZE, HOG_BLOCK_SIZE, HOG_BLOCK_STRIDE, HOG_CELL_SIZE, HOG_NBINS
)


def extract_hog(bgr_img: np.ndarray) -> np.ndarray:
    """Resize to 64×64 and extract HOG descriptor."""
    gray = cv2.cvtColor(cv2.resize(bgr_img, HOG_WIN_SIZE), cv2.COLOR_BGR2GRAY)
    # HOGDescriptor expects uint8
    return _hog.compute(gray).flatten()


def extract_hog_batch(X_rgb: np.ndarray) -> np.ndarray:
    """
    X_rgb : (N, H, W, 3) float32 0-1, RGB order
    Returns (N, feature_dim) float32
    """
    feats = []
    for img in X_rgb:
        bgr = (img[:, :, ::-1] * 255).astype(np.uint8)
        feats.append(extract_hog(bgr))
    return np.array(feats, dtype=np.float32)


def load_hog_dataset(
    dataset_root: str,
    task: str,
    split: str,
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract HOG features directly from disk one image at a time.
    Avoids loading the full (N, 224, 224, 3) image array into RAM —
    peak usage is one image at a time instead of the whole dataset.

    Returns
    -------
    X : np.ndarray  shape (N, feature_dim), dtype float32
    y : np.ndarray  shape (N,),             dtype int64
    """
    from pathlib import Path
    from ml.dataset import GESTURE_LABEL, FATIGUE_LABEL

    label_map = GESTURE_LABEL if task == "gesture" else FATIGUE_LABEL
    img_dir   = Path(dataset_root) / split / "images" / task

    feats, labels = [], []
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

            variants = [img]
            if augment:
                sized = cv2.resize(img, (224, 224))
                h, w  = sized.shape[:2]
                cx, cy = w // 2, h // 2
                variants += [
                    cv2.flip(sized, 1),
                    np.clip(sized.astype(np.int16) - 30, 0, 255).astype(np.uint8),
                    np.clip(sized.astype(np.int16) + 30, 0, 255).astype(np.uint8),
                    cv2.warpAffine(sized, cv2.getRotationMatrix2D((cx, cy), -10, 1.0), (w, h)),
                    cv2.warpAffine(sized, cv2.getRotationMatrix2D((cx, cy),  10, 1.0), (w, h)),
                    cv2.GaussianBlur(sized, (5, 5), 0),
                ]

            for v in variants:
                feats.append(extract_hog(v))
                labels.append(label_idx)

    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)


# ── Classifier wrapper ────────────────────────────────────────────────────────
class ClassicalClassifier:
    """
    Wraps either an SVM or Random Forest trained on HOG features.
    """

    def __init__(self, model_path: str, class_names: list[str]):
        self.model_path  = model_path
        self.class_names = class_names
        self._model      = None

    def is_available(self) -> bool:
        return os.path.exists(self.model_path)

    def _load(self):
        with open(self.model_path, "rb") as f:
            self._model = pickle.load(f)

    def predict(self, bgr_frame: np.ndarray) -> tuple[str, float]:
        """
        Returns (class_name, confidence).
        Confidence = max class probability (SVM w/ probability=True, RF always).
        """
        if self._model is None:
            self._load()

        feat = extract_hog(bgr_frame).reshape(1, -1)
        idx  = int(self._model.predict(feat)[0])

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(feat)[0]
            conf  = float(proba[idx])
        else:
            conf = 1.0

        return self.class_names[idx], conf

    def predict_proba(self, bgr_frame: np.ndarray) -> np.ndarray:
        if self._model is None:
            self._load()

        feat = extract_hog(bgr_frame).reshape(1, -1)

        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(feat)[0]

        idx = int(self._model.predict(feat)[0])
        proba = np.zeros(len(self.class_names))
        proba[idx] = 1.0
        return proba


# ── Training helpers ──────────────────────────────────────────────────────────
def train_svm(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                    probability=True, class_weight="balanced")),
    ])
    clf.fit(X_train, y_train)
    return clf


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf


def save_model(clf, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
