from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


FEATURE_VERSION = 1
HISTORY_OFFSETS_SEC = (1.5, 1.0, 0.5, 0.25, 0.0)
MIN_KEYPOINT_SCORE = 0.06
SNAPSHOT_FEATURES = 62
FEATURE_COUNT = SNAPSHOT_FEATURES * len(HISTORY_OFFSETS_SEC) + 1

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


def _midpoint(points: np.ndarray, left: int, right: int) -> tuple[float, float]:
    valid = []
    for index in (left, right):
        if points[index, 2] >= MIN_KEYPOINT_SCORE:
            valid.append(points[index, :2])
    if not valid:
        return 0.0, 0.0
    mean = np.mean(valid, axis=0)
    return float(mean[0]), float(mean[1])


def _snapshot_features(keypoints: np.ndarray, pose_score: float) -> np.ndarray:
    result = np.zeros(SNAPSHOT_FEATURES, dtype=np.float32)
    points = np.asarray(keypoints, dtype=np.float32).reshape(17, 3)
    valid = points[:, 2] >= MIN_KEYPOINT_SCORE
    if int(np.sum(valid)) < 4 or pose_score <= 0.0:
        return result

    ys = points[valid, 0]
    xs = points[valid, 1]
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    height = max(1e-4, ymax - ymin)
    width = max(1e-4, xmax - xmin)
    center_y = (ymin + ymax) * 0.5
    center_x = (xmin + xmax) * 0.5
    ratio = width / height

    shoulder_y, shoulder_x = _midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_y, hip_x = _midpoint(points, LEFT_HIP, RIGHT_HIP)
    torso_dy = abs(shoulder_y - hip_y)
    torso_dx = abs(shoulder_x - hip_x)
    torso_angle = float(np.arctan2(torso_dy, torso_dx + 1e-6) / np.pi)

    result[:11] = (
        1.0,
        float(pose_score),
        float(np.mean(valid)),
        center_y,
        center_x,
        height,
        width,
        float(np.clip(ratio, 0.0, 10.0)),
        shoulder_y,
        hip_y,
        torso_angle,
    )
    cursor = 11
    result[cursor : cursor + 17] = np.clip(
        (points[:, 0] - center_y) / height, -3.0, 3.0
    )
    cursor += 17
    result[cursor : cursor + 17] = np.clip(
        (points[:, 1] - center_x) / height, -3.0, 3.0
    )
    cursor += 17
    result[cursor : cursor + 17] = np.clip(points[:, 2], 0.0, 1.0)
    return result


def feature_at_frame(
    keypoints: np.ndarray,
    pose_scores: np.ndarray,
    fps: float,
    frame_index: int,
) -> np.ndarray:
    snapshots = []
    for offset in HISTORY_OFFSETS_SEC:
        index = max(0, frame_index - int(round(offset * fps)))
        snapshots.append(_snapshot_features(keypoints[index], float(pose_scores[index])))
    coverage = min(1.0, frame_index / max(1.0, HISTORY_OFFSETS_SEC[0] * fps))
    return np.concatenate((*snapshots, np.asarray([coverage], dtype=np.float32)))


def features_for_indices(
    keypoints: np.ndarray,
    pose_scores: np.ndarray,
    fps: float,
    indices: Iterable[int],
) -> np.ndarray:
    rows = [feature_at_frame(keypoints, pose_scores, fps, int(index)) for index in indices]
    if not rows:
        return np.empty((0, FEATURE_COUNT), dtype=np.float32)
    return np.stack(rows).astype(np.float32, copy=False)


def feature_from_history(
    history: Sequence[tuple[float, np.ndarray, float]], now: float
) -> np.ndarray:
    if not history:
        return np.zeros(FEATURE_COUNT, dtype=np.float32)
    times = np.asarray([item[0] for item in history], dtype=np.float64)
    snapshots = []
    for offset in HISTORY_OFFSETS_SEC:
        target = now - offset
        candidates = np.flatnonzero(times <= target)
        index = int(candidates[-1]) if len(candidates) else 0
        _, keypoints, score = history[index]
        snapshots.append(_snapshot_features(keypoints, float(score)))
    coverage = min(1.0, max(0.0, now - float(times[0])) / HISTORY_OFFSETS_SEC[0])
    return np.concatenate((*snapshots, np.asarray([coverage], dtype=np.float32)))
