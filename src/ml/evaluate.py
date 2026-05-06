"""
Evaluation script: compare Classical (HOG+SVM/RF) vs Modern (CNN) on the test set.
Produces a side-by-side accuracy/F1 table for the report.

Usage:
    python src/ml/evaluate.py --task gesture
    python src/ml/evaluate.py --task fatigue
    python src/ml/evaluate.py --task both
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_ROOT  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "CompVision_DataSet")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ml")


def _cnn_probabilities(task: str, X_rgb: np.ndarray):
    """
    Run CNN on (N,224,224,3) float32 RGB images.
    Returns (probs, preds) where probs is (N,C) float32 and preds is (N,) int,
    or (None, None) if the model is missing or incompatible.
    """
    import torch
    from ml.models import build_gesture_model, build_fatigue_model

    model_path = os.path.join(MODELS_DIR, f"cnn_{task}.pt")
    if not os.path.exists(model_path):
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_gesture_model() if task == "gesture" else build_fatigue_model()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except RuntimeError:
        print(f"  [CNN] Stale model at '{model_path}' (wrong class count). "
              f"Retrain with train.py.")
        return None, None
    model.eval()
    model.to(device)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X_rgb), 32):
            xb = torch.tensor(X_rgb[i:i+32]).permute(0, 3, 1, 2)
            xb = (xb - mean) / std
            xb = xb.to(device)
            probs_list.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    probs = np.concatenate(probs_list)
    return probs, probs.argmax(axis=1)


def _classical_predictions(task: str, clf_type: str, hog_feats: np.ndarray):
    """
    Run a pre-trained classical classifier on pre-computed HOG features.
    Returns (probs, preds) or (None, None) if the model is missing or incompatible.
    """
    from ml.classical import load_model

    model_path = os.path.join(MODELS_DIR, f"{clf_type}_{task}.pkl")
    if not os.path.exists(model_path):
        return None, None

    clf = load_model(model_path)
    try:
        probs = clf.predict_proba(hog_feats)
    except Exception:
        print(f"  [Classical] Stale model at '{model_path}' (wrong class count). "
              f"Retrain with train_classical.py.")
        return None, None
    return probs, probs.argmax(axis=1)


def evaluate_task(task: str):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from ml.dataset   import load_image_dataset, get_class_names
    from ml.classical import extract_hog_batch

    print(f"\n{'='*60}")
    print(f"  Evaluation — task: {task.upper()}")
    print(f"{'='*60}")

    X_test, y_test = load_image_dataset(DATA_ROOT, task, "test", augment=False)
    class_names    = get_class_names(task)
    print(f"Test samples : {len(y_test)}")
    print(f"Classes      : {class_names}\n")

    hog_feats = extract_hog_batch(X_test)   # computed once; reused for all classifiers
    results   = {}

    # ── Classical: SVM ────────────────────────────────────────────────────────
    svm_prob, preds_svm = _classical_predictions(task, "svm", hog_feats)
    if preds_svm is not None:
        acc = accuracy_score(y_test, preds_svm)
        f1  = f1_score(y_test, preds_svm, average="weighted", zero_division=0)
        results["HOG + SVM"] = (acc, f1, preds_svm)
        print(f"HOG + SVM     acc={acc:.3f}  f1={f1:.3f}")
    else:
        print("HOG + SVM     [model not found or stale — run train_classical.py first]")

    # ── Classical: Random Forest ──────────────────────────────────────────────
    _, preds_rf = _classical_predictions(task, "rf", hog_feats)
    if preds_rf is not None:
        acc = accuracy_score(y_test, preds_rf)
        f1  = f1_score(y_test, preds_rf, average="weighted", zero_division=0)
        results["HOG + RF"] = (acc, f1, preds_rf)
        print(f"HOG + RF      acc={acc:.3f}  f1={f1:.3f}")
    else:
        print("HOG + RF      [model not found or stale — run train_classical.py first]")

    # ── Modern: CNN ───────────────────────────────────────────────────────────
    cnn_prob, preds_cnn = _cnn_probabilities(task, X_test)
    if preds_cnn is not None:
        acc = accuracy_score(y_test, preds_cnn)
        f1  = f1_score(y_test, preds_cnn, average="weighted", zero_division=0)
        results["MobileNetV2 CNN"] = (acc, f1, preds_cnn)
        print(f"MobileNetV2   acc={acc:.3f}  f1={f1:.3f}")
    else:
        print("MobileNetV2   [model not found or stale — run train.py first]")

    # ── Hybrid: average probability ensemble (SVM + CNN) ─────────────────────
    if svm_prob is not None and cnn_prob is not None:
        if svm_prob.shape[1] != cnn_prob.shape[1]:
            print("Hybrid SVM+CNN [skipped — class count mismatch, retrain both models]")
        else:
            # Task-adaptive weights: SVM outperforms CNN on gesture (small dataset);
            # CNN outperforms SVM on fatigue (complex appearance classes).
            svm_w = 0.7 if task == "gesture" else 0.4
            ensemble_prob = svm_w * svm_prob + (1.0 - svm_w) * cnn_prob
            preds_hybrid  = ensemble_prob.argmax(axis=1)
            acc = accuracy_score(y_test, preds_hybrid)
            f1  = f1_score(y_test, preds_hybrid, average="weighted", zero_division=0)
            results["Hybrid (SVM+CNN)"] = (acc, f1, preds_hybrid)
            print(f"Hybrid SVM+CNN acc={acc:.3f}  f1={f1:.3f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*45}")
    print(f"{'Method':<22}  {'Accuracy':>8}  {'F1 (weighted)':>13}")
    print(f"{'─'*45}")
    for name, (acc, f1, _) in results.items():
        print(f"{name:<22}  {acc:>8.3f}  {f1:>13.3f}")
    print(f"{'─'*45}")

    # ── Detailed report for best method ──────────────────────────────────────
    if results:
        best_name  = max(results, key=lambda k: results[k][0])
        best_preds = results[best_name][2]
        print(f"\nDetailed report for best method ({best_name}):")
        print(classification_report(y_test, best_preds,
                                    target_names=class_names, zero_division=0))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare all classifiers")
    parser.add_argument("--task", choices=["gesture", "fatigue", "both"], default="both")
    args = parser.parse_args()

    tasks = ["gesture", "fatigue"] if args.task == "both" else [args.task]
    for task in tasks:
        evaluate_task(task)


if __name__ == "__main__":
    main()
