"""
MobileNetV2-based CNN classifiers for gesture and fatigue detection.
Uses transfer learning: pretrained ImageNet weights + custom top layers.
Works with plain NumPy/OpenCV — no PyTorch/TF dependency at inference time
once the model is saved; training requires torch + torchvision.
"""

import os
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
GESTURE_NUM_CLASSES = 3   # open_palm, thumbs_up, wrong_gesture
FATIGUE_NUM_CLASSES = 5   # alert, eyes_closed, head_pose,
                          # tired_expression, yawn
IMG_SIZE = 224


def build_mobilenet(num_classes: int, freeze_base: bool = True):
    """
    Build a MobileNetV2 classifier with a custom head.

    Parameters
    ----------
    num_classes  : int   Number of output classes.
    freeze_base  : bool  Freeze backbone weights (good for small datasets).

    Returns
    -------
    model : torch.nn.Module
    """
    import torch
    import torch.nn as nn
    from torchvision import models

    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_base:
        for param in backbone.parameters():
            param.requires_grad = False

    # Unfreeze the last 2 conv blocks for fine-tuning
    for layer in list(backbone.features.children())[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes),
    )

    return backbone


def build_gesture_model():
    return build_mobilenet(GESTURE_NUM_CLASSES)


def build_fatigue_model():
    return build_mobilenet(FATIGUE_NUM_CLASSES)


# ── Inference wrapper (no torch import at init) ───────────────────────────────
class CNNClassifier:
    """
    Thin wrapper around a saved .pt model for single-frame inference.
    Lazy-loads torch only when needed.
    """

    def __init__(self, model_path: str, num_classes: int, class_names: list[str]):
        self.model_path   = model_path
        self.num_classes  = num_classes
        self.class_names  = class_names
        self._model       = None
        self._device      = None

    def _load(self):
        import torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_mobilenet(self.num_classes, freeze_base=False)
        try:
            model.load_state_dict(
                torch.load(self.model_path, map_location=self._device, weights_only=True)
            )
        except RuntimeError:
            print(f"  [CNNClassifier] Stale model at '{self.model_path}' "
                  f"(wrong class count). Retrain with train.py.")
            return
        model.eval()
        model.to(self._device)
        self._model = model

    def is_available(self) -> bool:
        return os.path.exists(self.model_path)

    def predict(self, bgr_frame: np.ndarray) -> tuple[str, float]:
        """
        Classify a single BGR frame.

        Returns
        -------
        (class_name, confidence)  e.g. ("thumbs_up", 0.93)
        """
        import torch
        from torchvision import transforms

        if self._model is None:
            self._load()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # BGR → RGB
        rgb = bgr_frame[:, :, ::-1].copy()
        tensor = preprocess(rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        return self.class_names[idx], float(probs[idx])

    def predict_proba(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return full probability vector (one value per class)."""
        import torch
        from torchvision import transforms

        if self._model is None:
            self._load()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        rgb = bgr_frame[:, :, ::-1].copy()
        tensor = preprocess(rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return probs
