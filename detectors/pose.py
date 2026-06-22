from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

NOSE = 0
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

SKELETON_EDGES = (
    (NOSE, LEFT_SHOULDER),
    (NOSE, RIGHT_SHOULDER),
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
)

@dataclass(frozen=True)
class Pose:
    keypoints: np.ndarray
    bbox: Tuple[float, float, float, float]
    score: float

def pose_bbox_from_keypoints(
    keypoints: np.ndarray,
    min_score: float,
) -> Optional[Tuple[float, float, float, float]]:
    points = np.asarray(keypoints)
    valid = points[:, 2] >= min_score
    if int(np.sum(valid)) < 4:
        return None
    ys = points[valid, 0]
    xs = points[valid, 1]
    return float(ys.min()), float(xs.min()), float(ys.max()), float(xs.max())