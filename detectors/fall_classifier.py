from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .keypoint_features import FEATURE_COUNT, FEATURE_VERSION, feature_from_history
from .pose import Pose, pose_bbox_from_keypoints


@dataclass(frozen=True)
class ClassifierConfig:
    """Runtime-only pose filtering and alarm behavior."""

    min_pose_score: float = 0.05
    min_kpt_score: float = 0.06
    min_valid_keypoints: int = 4
    min_body_area: float = 0.0
    stop_when_multiple_people: bool = True
    multi_person_confirm_frames: int = 2
    alarm_hold_sec: float = 5.0


@dataclass(frozen=True)
class ClassifierState:
    probability: float
    consecutive: int
    threshold: float
    status: str
    triggered: bool


class ForestArtifact:
    """Small, dependency-free evaluator for the exported sklearn forest."""

    def __init__(self, path: Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or payload.get("classifier") != "forest":
            raise ValueError(f"Unsupported fall classifier artifact: {path}")
        if payload.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"Feature version mismatch: model={payload.get('feature_version')} "
                f"runtime={FEATURE_VERSION}"
            )

        self.name = str(payload.get("name", "forest"))
        self.threshold = float(payload["threshold"])
        self.confirmations = int(payload["confirmations"])
        self.trees = payload["trees"]
        if not self.trees:
            raise ValueError("Fall classifier contains no trees")

    def predict_probability(self, features: np.ndarray) -> float:
        row = np.asarray(features, dtype=np.float32).reshape(-1)
        if row.size != FEATURE_COUNT:
            raise ValueError(f"Expected {FEATURE_COUNT} features, got {row.size}")

        total = 0.0
        for tree in self.trees:
            node = 0
            while int(tree["feature"][node]) >= 0:
                feature = int(tree["feature"][node])
                node = int(
                    tree["left"][node]
                    if row[feature] <= float(tree["threshold"][node])
                    else tree["right"][node]
                )
            total += float(tree["positive_probability"][node])
        return total / len(self.trees)


class KeypointFallClassifier:
    """Stateful temporal fall classifier shared by Windows and Pi runtimes."""

    def __init__(self, model_path: Path, config: ClassifierConfig = ClassifierConfig()) -> None:
        self.model = ForestArtifact(model_path)
        self.config = config
        self.history: List[tuple[float, np.ndarray, float]] = []
        self.consecutive = 0

    def accepted_poses(self, poses: Sequence[Pose]) -> List[Pose]:
        accepted: List[Pose] = []
        for pose in poses:
            points = np.asarray(pose.keypoints, dtype=np.float32)
            valid = points[:, 2] >= self.config.min_kpt_score
            if pose.score < self.config.min_pose_score:
                continue
            if int(np.sum(valid)) < self.config.min_valid_keypoints:
                continue
            bbox = pose_bbox_from_keypoints(points, self.config.min_kpt_score)
            if bbox is None:
                continue
            ymin, xmin, ymax, xmax = bbox
            if (ymax - ymin) * (xmax - xmin) < self.config.min_body_area:
                continue
            accepted.append(pose)
        return sorted(accepted, key=lambda item: item.score, reverse=True)

    def update(self, pose: Optional[Pose], now: float) -> ClassifierState:
        if pose is None:
            keypoints = np.zeros((17, 3), dtype=np.float32)
            pose_score = 0.0
        else:
            keypoints = np.asarray(pose.keypoints, dtype=np.float32)
            pose_score = float(pose.score)

        self.history.append((float(now), keypoints, pose_score))
        cutoff = float(now) - 2.0
        while len(self.history) > 1 and self.history[1][0] < cutoff:
            self.history.pop(0)

        features = feature_from_history(self.history, float(now))
        probability = self.model.predict_probability(features)
        positive = pose is not None and probability >= self.model.threshold
        self.consecutive = self.consecutive + 1 if positive else 0
        triggered = self.consecutive >= self.model.confirmations

        if triggered:
            status = "FALL"
        elif positive:
            status = "POSSIBLE_FALL"
        else:
            status = "OK"
        return ClassifierState(
            probability=probability,
            consecutive=self.consecutive,
            threshold=self.model.threshold,
            status=status,
            triggered=triggered,
        )

    def reset(self) -> None:
        self.history.clear()
        self.consecutive = 0
