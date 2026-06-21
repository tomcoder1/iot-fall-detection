from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .keypoint_features import FEATURE_COUNTS, FEATURE_VERSION, feature_from_history
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
    votes: int
    threshold: float
    status: str
    triggered: bool


class ForestArtifact:
    """Small, dependency-free evaluator for the exported sklearn forest."""

    def __init__(self, path: Path, expected_platform: Optional[str] = None) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") not in (1, 2) or payload.get("classifier") != "forest":
            raise ValueError(f"Unsupported fall classifier artifact: {path}")
        if payload.get("feature_version") not in FEATURE_COUNTS:
            raise ValueError(
                f"Unsupported model feature version: {payload.get('feature_version')}"
            )

        self.name = str(payload.get("name", "forest"))
        self.feature_version = int(payload["feature_version"])
        self.platform = payload.get("platform")
        if expected_platform and self.platform and self.platform != expected_platform:
            raise ValueError(
                f"Wrong classifier for {expected_platform}: model is for {self.platform}"
            )
        self.threshold = float(payload["threshold"])
        if payload["format_version"] == 1:
            self.vote_window = int(payload["confirmations"])
            self.required_votes = int(payload["confirmations"])
        else:
            self.vote_window = int(payload["vote_window"])
            self.required_votes = int(payload["required_votes"])
        self.trees = payload["trees"]
        if not self.trees:
            raise ValueError("Fall classifier contains no trees")

    def predict_probability(self, features: np.ndarray) -> float:
        row = np.asarray(features, dtype=np.float32).reshape(-1)
        expected_count = FEATURE_COUNTS[self.feature_version]
        if row.size != expected_count:
            raise ValueError(f"Expected {expected_count} features, got {row.size}")

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

    def __init__(
        self,
        model_path: Path,
        config: ClassifierConfig = ClassifierConfig(),
        expected_platform: Optional[str] = None,
    ) -> None:
        self.model = ForestArtifact(model_path, expected_platform)
        self.config = config
        self.history: List[tuple[float, np.ndarray, float]] = []
        self.recent_votes: List[int] = []

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

        features = feature_from_history(
            self.history, float(now), self.model.feature_version
        )
        probability = self.model.predict_probability(features)
        # A pose often disappears for a few frames when the person reaches the
        # floor. The temporal feature history still carries valid fall evidence,
        # and training/tuning includes these missing-pose frames.
        positive = probability >= self.model.threshold
        self.recent_votes.append(int(positive))
        if len(self.recent_votes) > self.model.vote_window:
            self.recent_votes.pop(0)
        evidence = sum(self.recent_votes)
        triggered = evidence >= self.model.required_votes

        if triggered:
            status = "FALL"
        elif evidence:
            status = "POSSIBLE_FALL"
        else:
            status = "OK"
        return ClassifierState(
            probability=probability,
            votes=evidence,
            threshold=self.model.threshold,
            status=status,
            triggered=triggered,
        )

    def reset(self) -> None:
        self.history.clear()
        self.recent_votes.clear()
