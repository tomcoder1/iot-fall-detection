from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


SKELETON_EDGES = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]


@dataclass
class Person:
    keypoints: np.ndarray
    score: float
    bbox: Tuple[float, float, float, float]

    @property
    def valid_keypoints(self) -> int:
        return int(np.sum(self.keypoints[:, 2] > 0.25))

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.bbox
        return max(1.0, x2 - x1)

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.bbox
        return max(1.0, y2 - y1)


def make_empty_keypoints() -> np.ndarray:
    return np.zeros((17, 3), dtype=np.float32)


def keypoint(person: Person, index: int, min_score: float = 0.25) -> Optional[Tuple[float, float]]:
    x, y, score = person.keypoints[index]
    if score < min_score:
        return None
    return float(x), float(y)


def midpoint(
    person: Person,
    a: int,
    b: int,
    min_score: float = 0.25,
) -> Optional[Tuple[float, float]]:
    pa = keypoint(person, a, min_score)
    pb = keypoint(person, b, min_score)

    if pa is None and pb is None:
        return None
    if pa is None:
        return pb
    if pb is None:
        return pa

    return (pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5


def bbox_from_keypoints(
    keypoints: np.ndarray,
    frame_width: int,
    frame_height: int,
    min_score: float = 0.25,
) -> Tuple[float, float, float, float]:
    valid = keypoints[:, 2] >= min_score

    if not np.any(valid):
        return 0.0, 0.0, 1.0, 1.0

    xs = keypoints[valid, 0]
    ys = keypoints[valid, 1]

    x1 = float(np.clip(np.min(xs), 0, frame_width - 1))
    y1 = float(np.clip(np.min(ys), 0, frame_height - 1))
    x2 = float(np.clip(np.max(xs), 0, frame_width - 1))
    y2 = float(np.clip(np.max(ys), 0, frame_height - 1))

    pad_x = max(8.0, (x2 - x1) * 0.12)
    pad_y = max(8.0, (y2 - y1) * 0.12)

    x1 = float(np.clip(x1 - pad_x, 0, frame_width - 1))
    y1 = float(np.clip(y1 - pad_y, 0, frame_height - 1))
    x2 = float(np.clip(x2 + pad_x, 0, frame_width - 1))
    y2 = float(np.clip(y2 + pad_y, 0, frame_height - 1))

    return x1, y1, x2, y2