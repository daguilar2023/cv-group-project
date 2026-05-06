"""
Hybrid detector: combines classical (HOG+SVM) and modern (CNN) pipelines at runtime.

For gesture classification:
  - Classical: HOG + SVM (rule-based fallback if model missing)
  - Modern:    MobileNetV2 CNN
  - Hybrid:    weighted probability ensemble (CNN 60%, SVM 40%)

For fatigue classification:
  - Classical: MediaPipe EAR/MAR/head-pose heuristics (classmate's code)
  - Modern:    MobileNetV2 CNN on full face crop
  - Hybrid:    CNN adds a second opinion; if CNN and heuristic agree → high confidence;
               if they disagree → show both, trust CNN for visual cues
               (eyes_closed, yawn, tired_expression) and heuristic for geometry
               (head_down, head_tilt_left, head_tilt_right)
"""

import os
from collections import deque
import numpy as np

# Paths resolved relative to this file
_ML_DIR    = os.path.dirname(__file__)
MODELS_DIR = os.path.join(_ML_DIR, "..", "..", "models", "ml")

# ── Gesture classes ────────────────────────────────────────────────────────────
from ml.dataset import GESTURE_CLASSES, FATIGUE_CLASSES

# Fatigue classes whose detection is better with CNN (appearance-based)
CNN_DOMINANT_FATIGUE = {"eyes_closed", "yawn"}
# Fatigue classes that require heuristic corroboration before CNN alert fires
# (either too subtle for single-frame CNN, or geometry-based)
HEURISTIC_DOMINANT_FATIGUE = {"head_pose", "tired_expression"}


class TemporalSmoother:
    """
    Smooths CNN predictions over a rolling window of frames.
    Averages probability vectors so a single noisy frame can't trigger an alert.
    """

    def __init__(self, num_classes: int, class_names: list[str], window: int = 10):
        self.num_classes  = num_classes
        self.class_names  = class_names
        self.buffer       = deque(maxlen=window)

    def update(self, proba: np.ndarray) -> tuple[str, float]:
        """
        Add new probability vector, return (smoothed_class, smoothed_confidence).
        """
        self.buffer.append(proba)
        smoothed = np.mean(self.buffer, axis=0)
        idx  = int(np.argmax(smoothed))
        conf = float(smoothed[idx])
        return self.class_names[idx], conf

    def reset(self):
        self.buffer.clear()


class HybridGestureClassifier:
    """
    Drop-in replacement for GestureClassifier.
    Falls back to heuristic if CNN model is not available.
    """

    def __init__(self):
        from ml.models    import CNNClassifier
        from ml.classical import ClassicalClassifier
        from gesture_activation.classifier import GestureClassifier

        cnn_path = os.path.join(MODELS_DIR, "cnn_gesture.pt")
        svm_path = os.path.join(MODELS_DIR, "svm_gesture.pkl")

        self._cnn        = CNNClassifier(cnn_path, len(GESTURE_CLASSES), GESTURE_CLASSES)
        self._svm        = ClassicalClassifier(svm_path, GESTURE_CLASSES)
        self._heuristic  = GestureClassifier()

    @property
    def cnn_available(self) -> bool:
        return self._cnn.is_available()

    @property
    def svm_available(self) -> bool:
        return self._svm.is_available()

    def classify_hybrid(self, bgr_frame: np.ndarray) -> tuple[str, float, str]:
        """
        Returns (gesture_name, confidence, method_used).
        method_used is one of: 'cnn', 'svm', 'ensemble', 'heuristic'
        """
        cnn_ok = self.cnn_available
        svm_ok = self.svm_available

        if cnn_ok and svm_ok:
            cnn_prob = self._cnn.predict_proba(bgr_frame)
            svm_prob = self._svm.predict_proba(bgr_frame)
            # Weighted ensemble: CNN gets more weight
            ensemble = 0.6 * cnn_prob + 0.4 * svm_prob
            idx  = int(np.argmax(ensemble))
            conf = float(ensemble[idx])
            return GESTURE_CLASSES[idx], conf, "ensemble"

        if cnn_ok:
            name, conf = self._cnn.predict(bgr_frame)
            return name, conf, "cnn"

        if svm_ok:
            name, conf = self._svm.predict(bgr_frame)
            return name, conf, "svm"

        return "UNKNOWN", 0.0, "no_model"

    def classify(self, landmarks) -> str:
        """Heuristic-only path (used when no frame crop is available)."""
        return self._heuristic.classify(landmarks)


class HybridFatigueClassifier:
    """
    Wraps the CNN fatigue classifier for per-frame scoring.
    Intended to be used alongside (not replacing) the existing heuristic trackers.
    """

    def __init__(self):
        from ml.models import CNNClassifier

        cnn_path = os.path.join(MODELS_DIR, "cnn_fatigue.pt")
        self._cnn = CNNClassifier(cnn_path, len(FATIGUE_CLASSES), FATIGUE_CLASSES)

    @property
    def available(self) -> bool:
        return self._cnn.is_available()

    def predict(self, bgr_frame: np.ndarray) -> tuple[str, float]:
        """
        Returns (class_name, confidence) from CNN.
        class_name is one of FATIGUE_CLASSES.
        """
        if not self.available:
            return "alert", 0.0
        return self._cnn.predict(bgr_frame)

    def predict_proba(self, bgr_frame: np.ndarray) -> dict[str, float]:
        """Return {class_name: probability} dict."""
        if not self.available:
            return {c: 0.0 for c in FATIGUE_CLASSES}
        probs = self._cnn.predict_proba(bgr_frame)
        return dict(zip(FATIGUE_CLASSES, probs.tolist()))


def hybrid_fatigue_decision(
    heuristic_alerts: list[str],
    cnn_class: str,
    cnn_conf: float,
    cnn_threshold: float = 0.55,
    yawn_threshold: float = 0.70,   # stricter threshold for yawn — prone to false positives
) -> tuple[list[str], str]:
    """
    Merge heuristic alert list with CNN prediction.

    Parameters
    ----------
    heuristic_alerts : list of active heuristic alert strings
                       (e.g. ["DROWSY EYES", "YAWN"])
    cnn_class        : top CNN class name (e.g. "eyes_closed")
    cnn_conf         : CNN confidence (0-1)
    cnn_threshold    : minimum CNN confidence to act on its prediction

    Returns
    -------
    final_alerts : merged alert list (may be extended by CNN)
    source       : "heuristic", "cnn", or "both"
    """
    final = list(heuristic_alerts)
    added_by_cnn = False

    # Yawn requires stricter confidence to avoid false positives from talking/mouth movement
    effective_threshold = yawn_threshold if cnn_class == "yawn" else cnn_threshold

    if cnn_conf >= effective_threshold and cnn_class != "alert":
        # Map CNN class name → display alert string
        cnn_alert_map = {
            "eyes_closed":     "DROWSY EYES (CNN)",
            "yawn":            "YAWN (CNN)",
            "tired_expression":"FATIGUE EXPRESSION (CNN)",
            "head_pose":       "HEAD POSE (CNN)",
        }
        cnn_alert = cnn_alert_map.get(cnn_class)
        if cnn_alert and cnn_alert not in final:
            # Heuristic-dominant classes only fire when heuristics already agree
            if cnn_class in HEURISTIC_DOMINANT_FATIGUE and not heuristic_alerts:
                pass
            else:
                final.append(cnn_alert)
                added_by_cnn = True

    if not heuristic_alerts and added_by_cnn:
        source = "cnn"
    elif heuristic_alerts and added_by_cnn:
        source = "both"
    else:
        source = "heuristic"

    return final, source
