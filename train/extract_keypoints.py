from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from detectors.windows_movenet_multipose_fall import (
    INPUT_SIZE,
    NUM_THREADS,
    MoveNetMultiPose,
    resolve_model_path,
)
from train.dataset import cache_path, load_records


def _fps(capture: cv2.VideoCapture) -> float:
    value = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return value if np.isfinite(value) and value > 1.0 else 30.0


def extract_video(model: MoveNetMultiPose, path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = _fps(capture)
    keypoint_frames = []
    pose_scores = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            poses, _ = model.infer(frame)
            candidates = []
            for pose in poses:
                valid = int(np.sum(pose.keypoints[:, 2] >= 0.06))
                if pose.score >= 0.05 and valid >= 4:
                    candidates.append(pose)
            candidates.sort(key=lambda pose: pose.score, reverse=True)
            if candidates:
                keypoint_frames.append(candidates[0].keypoints.astype(np.float32))
                pose_scores.append(float(candidates[0].score))
            else:
                keypoint_frames.append(np.zeros((17, 3), dtype=np.float32))
                pose_scores.append(0.0)
    finally:
        capture.release()
    return (
        np.asarray(keypoint_frames, dtype=np.float32),
        np.asarray(pose_scores, dtype=np.float32),
        fps,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--cache", type=Path, default=Path("train/cache"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    records = load_records(args.dataset)
    model = MoveNetMultiPose(resolve_model_path(), NUM_THREADS, INPUT_SIZE)
    for number, record in enumerate(records, 1):
        output = cache_path(args.cache, record)
        if output.exists() and not args.force:
            print(f"[{number:03d}/{len(records)}] cached {record.path}")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        keypoints, pose_scores, fps = extract_video(model, record.path)
        np.savez_compressed(
            output,
            keypoints=keypoints,
            pose_scores=pose_scores,
            fps=np.float32(fps),
        )
        print(f"[{number:03d}/{len(records)}] {len(keypoints):4d} frames {record.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
