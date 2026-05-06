"""
Classical ML training script: HOG features + SVM and Random Forest.

Usage:
    python src/ml/train_classical.py --task gesture
    python src/ml/train_classical.py --task fatigue
    python src/ml/train_classical.py --task both
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_ROOT  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "CompVision_DataSet")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ml")


def evaluate(clf, X_feats: np.ndarray, y: np.ndarray, split_name: str):
    from sklearn.metrics import accuracy_score, classification_report
    preds = clf.predict(X_feats)
    acc   = accuracy_score(y, preds)
    print(f"  {split_name} accuracy: {acc:.3f}")
    return acc


def train_task(task: str, classifier_type: str = "svm"):
    from ml.dataset    import get_class_names
    from ml.classical  import (load_hog_dataset, train_svm,
                                train_random_forest, save_model)

    print(f"\n{'='*50}")
    print(f"Training Classical ({classifier_type.upper()}) — task: {task}")
    print(f"{'='*50}")

    class_names = get_class_names(task)
    print(f"Classes : {class_names}")

    print("Extracting HOG features...")
    X_tr_hog,  y_train = load_hog_dataset(DATA_ROOT, task, "train", augment=True)
    X_val_hog, y_val   = load_hog_dataset(DATA_ROOT, task, "val",   augment=False)
    X_te_hog,  y_test  = load_hog_dataset(DATA_ROOT, task, "test",  augment=False)
    print(f"  Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")
    print(f"  Feature dim: {X_tr_hog.shape[1]}")

    print(f"Fitting {classifier_type.upper()}...")
    if classifier_type == "svm":
        clf = train_svm(X_tr_hog, y_train)
    else:
        clf = train_random_forest(X_tr_hog, y_train)

    train_acc = evaluate(clf, X_tr_hog,  y_train, "Train")
    val_acc   = evaluate(clf, X_val_hog, y_val,   "Val  ")
    test_acc  = evaluate(clf, X_te_hog,  y_test,  "Test ")

    # Detailed report on test set
    from sklearn.metrics import classification_report
    preds = clf.predict(X_te_hog)
    print(f"\nTest classification report:\n"
          f"{classification_report(y_test, preds, target_names=class_names, zero_division=0)}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"{classifier_type}_{task}.pkl")
    save_model(clf, save_path)
    print(f"Model saved → {save_path}")
    return save_path, test_acc


def main():
    parser = argparse.ArgumentParser(description="Train classical HOG+SVM/RF classifiers")
    parser.add_argument("--task",       choices=["gesture", "fatigue", "both"],
                        default="both")
    parser.add_argument("--classifier", choices=["svm", "rf", "both"],
                        default="both",
                        help="svm = SVM, rf = Random Forest, both = train both")
    args = parser.parse_args()

    tasks       = ["gesture", "fatigue"] if args.task       == "both" else [args.task]
    classifiers = ["svm", "rf"]          if args.classifier == "both" else [args.classifier]

    for task in tasks:
        for clf_type in classifiers:
            train_task(task, clf_type)


if __name__ == "__main__":
    main()
