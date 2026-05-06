"""
CNN training script (MobileNetV2 with transfer learning).
Uses lazy disk-based loading so large datasets don't blow up RAM.

Usage:
    python ml/train.py --task gesture
    python ml/train.py --task fatigue
    python ml/train.py --task both
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "CompVision_DataSet"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "ml"))


# ── Lazy PyTorch Dataset (reads images from disk per batch) ───────────────────
def make_lazy_dataset(data_root: str, task: str, split: str, augment: bool):
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    from ml.dataset import GESTURE_LABEL, FATIGUE_LABEL

    label_map = GESTURE_LABEL if task == "gesture" else FATIGUE_LABEL
    img_dir   = Path(data_root) / split / "images" / task

    # Base transform (always applied)
    base_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Augmentation transform (only for training)
    aug_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    transform = aug_tf if augment else base_tf

    class _LazyDataset(Dataset):
        def __init__(self):
            self.samples = []   # list of (path, label)
            for class_name, label_idx in label_map.items():
                class_dir = img_dir / class_name
                if not class_dir.exists():
                    continue
                for p in sorted(class_dir.iterdir()):
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        self.samples.append((str(p), label_idx))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            import cv2
            path, label = self.samples[idx]
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return transform(img), label

    return _LazyDataset()


def train_task(task: str, epochs: int = 30, batch_size: int = 16, lr: float = 1e-3):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset

    from ml.dataset import get_class_names
    from ml.models  import build_gesture_model, build_fatigue_model

    print(f"\n{'='*50}")
    print(f"Training CNN — task: {task}")
    print(f"{'='*50}")

    train_ds = make_lazy_dataset(DATA_ROOT, task, "train", augment=True)
    val_ds   = make_lazy_dataset(DATA_ROOT, task, "val",   augment=False)

    class_names = get_class_names(task)
    num_classes = len(class_names)

    # Re-split val from train when val set is too small for reliable model selection
    # Gesture has far fewer images so a lower per-class threshold avoids sacrificing
    # too many training samples to validation.
    MIN_VAL_PER_CLASS = 5 if task == "gesture" else 15
    if len(val_ds) < MIN_VAL_PER_CLASS * num_classes:
        import random as _random
        _random.seed(42)
        _train_noaug = make_lazy_dataset(DATA_ROOT, task, "train", augment=False)
        buckets: dict = {}
        for i, (_, lbl) in enumerate(train_ds.samples):
            buckets.setdefault(lbl, []).append(i)
        val_idx, train_idx = [], []
        n_val = max(5, MIN_VAL_PER_CLASS)
        for lbl, idxs in sorted(buckets.items()):
            _random.shuffle(idxs)
            val_idx.extend(idxs[:n_val])
            train_idx.extend(idxs[n_val:])
        val_ds   = Subset(_train_noaug, val_idx)
        train_ds = Subset(train_ds, train_idx)
        print(f"  Val set too small — resampled from train: "
              f"train={len(train_ds)}  val={len(val_ds)}")

    print(f"Classes : {class_names}")
    print(f"Train   : {len(train_ds)} samples")
    print(f"Val     : {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")

    model = build_gesture_model() if task == "gesture" else build_fatigue_model()
    model.to(device)

    # Focal loss with inverse-frequency class weights
    samples_list   = (train_ds.dataset.samples if isinstance(train_ds, Subset)
                      else train_ds.samples)
    active_indices = (train_ds.indices if isinstance(train_ds, Subset)
                      else range(len(samples_list)))
    label_counts = np.zeros(num_classes, dtype=np.float32)
    for i in active_indices:
        label_counts[samples_list[i][1]] += 1
    label_counts = np.where(label_counts == 0, 1, label_counts)
    raw_weights  = 1.0 / label_counts
    weights = torch.tensor(raw_weights / raw_weights.mean(), dtype=torch.float32).to(device)

    class FocalLoss(nn.Module):
        def __init__(self, weight=None, gamma=2.0):
            super().__init__()
            self.weight = weight
            self.gamma  = gamma
        def forward(self, inputs, targets):
            ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
            pt = torch.exp(-ce)
            return (((1 - pt) ** self.gamma) * ce).mean()

    criterion = FocalLoss(weight=weights, gamma=2.0)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    best_val_acc = 0.0
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"cnn_{task}.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            n          += len(yb)

        train_acc  = correct / n
        train_loss = total_loss / n

        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                n       += len(yb)
        val_acc = correct / n
        scheduler.step(val_acc)

        print(f"Epoch {epoch:3d}/{epochs}  "
              f"loss={train_loss:.4f}  "
              f"train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f}) → {save_path}")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to   : {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Train CNN classifiers")
    parser.add_argument("--task",       choices=["gesture", "fatigue", "both"],
                        default="both")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    tasks = ["gesture", "fatigue"] if args.task == "both" else [args.task]
    for task in tasks:
        train_task(task, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
