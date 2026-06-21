from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2

from app_common import PoseModel
from detectors.fall_classifier import KeypointFallClassifier


VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv")


def find_videos(dataset_root: Path) -> List[Tuple[str, Path]]:
    videos: List[Tuple[str, Path]] = []
    for subject_dir in sorted(dataset_root.glob("Subject *")):
        for label in ("ADL", "Fall"):
            for pattern in VIDEO_EXTENSIONS:
                videos.extend((label, path) for path in sorted((subject_dir / label).glob(pattern)))
    return videos


def process_video(
    model: PoseModel,
    detector: KeypointFallClassifier,
    label: str,
    path: Path,
) -> Dict[str, object]:
    detector.reset()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 1.0:
        fps = 30.0

    frame_index = 0
    predicted_fall = False
    fall_frame = ""
    peak_probability = 0.0
    peak_votes = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_index += 1
            now = frame_index / fps
            poses, _ = model.infer(frame)
            accepted = detector.accepted_poses(poses)
            if accepted:
                state = detector.update(accepted[0], now)
            else:
                state = detector.update(None, now)
            peak_probability = max(peak_probability, state.probability)
            peak_votes = max(peak_votes, state.votes)
            if state.triggered and not predicted_fall:
                predicted_fall = True
                fall_frame = frame_index
    finally:
        cap.release()

    truth = "FALL" if label == "Fall" else "ADL"
    return {
        "path": str(path),
        "truth": truth,
        "pred": "FALL" if predicted_fall else "NO FALL",
        "frames": frame_index,
        "fps": fps,
        "fall_frame": fall_frame,
        "peak_probability": peak_probability,
        "peak_votes": peak_votes,
    }


def run_dataset_test(
    model: PoseModel,
    detector: KeypointFallClassifier,
    dataset_root: Path,
    csv_path: Path,
    title: str,
    video_keys: Optional[Set[str]] = None,
) -> int:
    videos = find_videos(dataset_root)
    if video_keys is not None:
        videos = [item for item in videos if item[1].as_posix() in video_keys]
    if not videos:
        raise RuntimeError(f"No videos found under {dataset_root.resolve()}")

    rows: List[Dict[str, object]] = []
    tp = fp = tn = fn = 0
    for index, (label, path) in enumerate(videos, 1):
        row = process_video(model, detector, label, path)
        rows.append(row)
        truth, pred = row["truth"], row["pred"]
        tp += int(truth == "FALL" and pred == "FALL")
        fp += int(truth == "ADL" and pred == "FALL")
        tn += int(truth == "ADL" and pred == "NO FALL")
        fn += int(truth == "FALL" and pred == "NO FALL")
        print(
            f"[{index:03d}/{len(videos)}] true={truth:<4} pred={pred:<7} "
            f"peak={float(row['peak_probability']):.3f} votes={row['peak_votes']}/"
            f"{detector.model.required_votes} fall_frame={row['fall_frame']} {path}"
        )

    total = tp + fp + tn + fn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    print(f"\n========== {title} ==========")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Accuracy: {(tp + tn) / max(1, total):.3f}")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f}")
    print(f"Specificity: {specificity:.3f} | F1: {f1:.3f}")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")
    return 0
