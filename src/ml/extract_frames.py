"""
Extract frames from all videos in the dataset and save them as JPEG images.
Extracted frames are added into the same images/ folder structure so they
automatically get picked up by dataset.py during training.

Usage:
    python ml/extract_frames.py                  # extract every 5th frame (default)
    python ml/extract_frames.py --every 3        # extract every 3rd frame
    python ml/extract_frames.py --split train    # only process train split
"""

import argparse
import os
import sys
import cv2
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "CompVision_DataSet")
)

# Map video subfolder names to image subfolder names
# (activation videos → gesture images)
VIDEO_TO_IMAGE_TASK = {
    "activation": "gesture",
    "fatigue":    "fatigue",
}

# Map activation video class folders → gesture image class folders
ACTIVATION_CLASS_MAP = {
    "correct_sequence": None,   # skip — multi-gesture sequence, not a single class
    "wrong_order":      None,   # skip
    "open_palm_only":   "open_palm",
    "thumbs_up_only":   "thumbs_up",
    "wrong_gesture":    "wrong_gesture",
}


def extract_from_video(
    video_path: Path,
    out_dir: Path,
    every_nth: int,
    prefix: str,
) -> int:
    """Extract every nth frame from video_path into out_dir. Returns frame count saved."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path.name}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    saved, frame_idx = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_nth == 0:
            fname = out_dir / f"{prefix}_f{frame_idx:04d}.jpeg"
            cv2.imwrite(str(fname), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def process_split(split: str, every_nth: int) -> dict[str, int]:
    counts = {}
    video_root = Path(DATA_ROOT) / split / "videos"

    if not video_root.exists():
        print(f"  [SKIP] {video_root} does not exist")
        return counts

    for video_task_dir in sorted(video_root.iterdir()):
        if not video_task_dir.is_dir():
            continue

        video_task = video_task_dir.name          # "activation" or "fatigue"
        image_task = VIDEO_TO_IMAGE_TASK.get(video_task)
        if image_task is None:
            continue

        for class_dir in sorted(video_task_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            # Determine destination image class folder
            if video_task == "activation":
                dest_class = ACTIVATION_CLASS_MAP.get(class_name)
                if dest_class is None:
                    print(f"  [SKIP] {split}/videos/{video_task}/{class_name} "
                          f"(multi-gesture, no single label)")
                    continue
            else:
                dest_class = class_name

            out_dir = (Path(DATA_ROOT) / split / "images" / image_task / dest_class)

            videos = sorted(f for f in class_dir.iterdir()
                            if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})

            for video_path in videos:
                prefix = f"vid_{video_path.stem}"
                n = extract_from_video(video_path, out_dir, every_nth, prefix)
                key = f"{split}/{image_task}/{dest_class}"
                counts[key] = counts.get(key, 0) + n
                print(f"  {split}/videos/{video_task}/{class_name}/{video_path.name}"
                      f" → {key}  (+{n} frames)")

    return counts


def main():
    parser = argparse.ArgumentParser(description="Extract frames from dataset videos")
    parser.add_argument("--every",  type=int, default=5,
                        help="Save every Nth frame (default: 5)")
    parser.add_argument("--split",  choices=["train", "val", "test", "all"],
                        default="all")
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    total  = {}

    for split in splits:
        print(f"\n── {split.upper()} ──────────────────────────────")
        counts = process_split(split, args.every)
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v

    print("\n── Summary ─────────────────────────────────")
    grand = 0
    for k, v in sorted(total.items()):
        print(f"  {k:<40s}  {v:>4d} frames")
        grand += v
    print(f"  {'TOTAL':<40s}  {grand:>4d} frames")


if __name__ == "__main__":
    main()
