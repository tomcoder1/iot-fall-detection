from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from detectors.fall_core import FallDetector
from detectors.pi4_coral_posenet_fall import (
    CONFIG,
    MODEL_PATH,
    PROJECT_POSENET_DIR,
    CoralPoseNet,
)

DATASET_ROOT = Path("dataset")
DEBUG_CSV_PATH = Path("fall_debug_results_pi.csv")
VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv")


def find_videos(dataset_root: Path) -> List[Tuple[str, Path]]:
    videos: List[Tuple[str, Path]] = []
    for subject_dir in sorted(dataset_root.glob("Subject *")):
        for label in ("ADL", "Fall"):
            label_dir = subject_dir / label
            if not label_dir.exists():
                continue
            for pattern in VIDEO_EXTENSIONS:
                for path in sorted(label_dir.glob(pattern)):
                    videos.append((label, path))
    return videos


def safe_video_fps(cap: cv2.VideoCapture) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 1.0:
        return 30.0
    return fps


def process_video(model: CoralPoseNet, label: str, path: Path) -> Dict[str, object]:
    detector = FallDetector(CONFIG)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = safe_video_fps(cap)
    frame_idx = 0
    multi_person_hits = 0
    predicted_fall = False
    fall_frame: Optional[int] = None

    max_hip_v = 0.0
    max_shoulder_v = 0.0
    max_ratio = 0.0
    min_angle: Optional[float] = None
    max_score = 0.0
    best_debug: Dict[str, object] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx += 1
            now = frame_idx / fps

            poses, _ = model.infer(frame)
            accepted = detector.accepted_poses(poses)
            people_count = len(accepted)

            if people_count > 1:
                multi_person_hits += 1
            else:
                multi_person_hits = 0

            multi_person_disabled = (
                CONFIG.stop_when_multiple_people
                and multi_person_hits >= CONFIG.multi_person_confirm_frames
            )

            if people_count == 0 or multi_person_disabled:
                detector.reset()
                continue

            results = detector.update(accepted[:1], now)

            for _, state in results:
                dbg = state.debug

                max_hip_v = max(max_hip_v, float(dbg.get("hip_speed", 0.0)))
                max_shoulder_v = max(max_shoulder_v, float(dbg.get("shoulder_speed", 0.0)))
                max_ratio = max(max_ratio, float(dbg.get("ratio", 0.0)))

                angle = float(dbg.get("angle", -1.0))
                if angle >= 0:
                    min_angle = angle if min_angle is None else min(min_angle, angle)

                score = 0.0
                if bool(dbg.get("recent_upright", False)):
                    score += 1.0
                if bool(dbg.get("strong_motion", False)):
                    score += 1.0
                if bool(dbg.get("horizontal", False)):
                    score += 1.0
                if bool(dbg.get("low_enough", False)):
                    score += 1.0
                score += min(
                    1.0,
                    float(dbg.get("max_down_speed", 0.0)) / max(CONFIG.fall_drop_speed, 1e-6),
                )

                if score > max_score:
                    max_score = score
                    best_debug = dict(dbg)
                    best_debug["frame"] = frame_idx
                    best_debug["status"] = state.last_status

                if state.last_status == "FALL" and not predicted_fall:
                    predicted_fall = True
                    fall_frame = frame_idx
    finally:
        cap.release()

    truth = "FALL" if label.lower() == "fall" else "ADL"
    pred = "FALL" if predicted_fall else "NO FALL"
    correct = (truth == "FALL" and predicted_fall) or (truth == "ADL" and not predicted_fall)

    return {
        "path": str(path),
        "truth": truth,
        "pred": pred,
        "correct": correct,
        "frames": frame_idx,
        "fps": fps,
        "fall_frame": fall_frame if fall_frame is not None else "",
        "hip_v": max_hip_v,
        "shoulder_v": max_shoulder_v,
        "ratio": max_ratio,
        "angle": min_angle if min_angle is not None else -1.0,
        "best_frame": best_debug.get("frame", ""),
        "best_status": best_debug.get("status", ""),
        "best_recent_upright": best_debug.get("recent_upright", ""),
        "best_strong_motion": best_debug.get("strong_motion", ""),
        "best_horizontal": best_debug.get("horizontal", ""),
        "best_low_enough": best_debug.get("low_enough", ""),
    }


def main() -> int:
    videos = find_videos(DATASET_ROOT)
    if not videos:
        raise RuntimeError(f"No videos found under {DATASET_ROOT.resolve()}")

    print(f"Found {len(videos)} videos.")
    print(f"Loading Coral PoseNet: {MODEL_PATH.resolve()}")

    model = CoralPoseNet(MODEL_PATH, PROJECT_POSENET_DIR)

    rows: List[Dict[str, object]] = []
    tp = fp = tn = fn = 0

    for idx, (label, path) in enumerate(videos, start=1):
        row = process_video(model, label, path)
        rows.append(row)

        truth = row["truth"]
        pred = row["pred"]

        if truth == "FALL" and pred == "FALL":
            tp += 1
        elif truth == "ADL" and pred == "FALL":
            fp += 1
        elif truth == "ADL" and pred == "NO FALL":
            tn += 1
        elif truth == "FALL" and pred == "NO FALL":
            fn += 1

        print(
            f"[{idx:03d}/{len(videos)}] true={truth:<4} pred={pred:<7} "
            f"frames={int(row['frames']):4d} hip_v={float(row['hip_v']):.2f} "
            f"shoulder_v={float(row['shoulder_v']):.2f} ratio={float(row['ratio']):.2f} "
            f"angle={float(row['angle']):.0f} fall_frame={row['fall_frame']} {row['path']}"
        )

    total = max(1, tp + fp + tn + fn)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)

    print("\n========== PI CORAL RESULTS ==========")
    print(f"Total videos: {total}")
    print("\nConfusion matrix:")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    print(f"\nAccuracy:    {accuracy:.3f}")
    print(f"Precision:   {precision:.3f}")
    print(f"Recall:      {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 score:    {f1:.3f}")

    print("\nFalse positives:")
    for r in rows:
        if r["truth"] == "ADL" and r["pred"] == "FALL":
            print(" FP ", r["path"])

    print("\nFalse negatives:")
    for r in rows:
        if r["truth"] == "FALL" and r["pred"] == "NO FALL":
            print(
                " FN ",
                r["path"],
                f"best_frame={r['best_frame']} status={r['best_status']} "
                f"upright={r['best_recent_upright']} motion={r['best_strong_motion']} "
                f"horizontal={r['best_horizontal']} low={r['best_low_enough']}",
            )

    with DEBUG_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote debug CSV: {DEBUG_CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())