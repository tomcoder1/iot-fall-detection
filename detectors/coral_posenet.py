from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image

from pose_types import Person, bbox_from_keypoints, make_empty_keypoints


CORAL_KEYPOINT_TO_INDEX = {
    "nose": 0,
    "left eye": 1,
    "right eye": 2,
    "left ear": 3,
    "right ear": 4,
    "left shoulder": 5,
    "right shoulder": 6,
    "left elbow": 7,
    "right elbow": 8,
    "left wrist": 9,
    "right wrist": 10,
    "left hip": 11,
    "right hip": 12,
    "left knee": 13,
    "right knee": 14,
    "left ankle": 15,
    "right ankle": 16,
}


class CoralPoseNetDetector:
    model_name = "Coral PoseNet"

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        try:
            from pose_engine import PoseEngine
        except ImportError as exc:
            raise RuntimeError(
                "Missing pose_engine.py. Put pose_engine.py beside main_pi.py."
            ) from exc

        self.engine = PoseEngine(str(model_path))

        try:
            shape = self.engine.get_input_tensor_shape()
            self.input_height = int(shape[1])
            self.input_width = int(shape[2])
        except Exception:
            self.input_height = 481
            self.input_width = 641

        print(f"Loaded {self.model_name}")
        print(f"PoseNet input size: {self.input_width}x{self.input_height}")

    def detect(self, frame: np.ndarray) -> list[Person]:
        frame_height, frame_width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        detected = self.engine.DetectPosesInImage(pil_image)

        if isinstance(detected, tuple):
            poses = detected[0]
        else:
            poses = detected

        people: list[Person] = []

        for pose in poses:
            keypoints = make_empty_keypoints()
            pose_score = float(getattr(pose, "score", 1.0))

            for label, keypoint in pose.keypoints.items():
                index = self._label_to_index(label)

                if index is None:
                    continue

                point = getattr(keypoint, "point", None)

                if point is None:
                    continue

                raw_x, raw_y = self._point_to_xy(point)
                x, y = self._map_point_to_frame(raw_x, raw_y, frame_width, frame_height)

                score = float(getattr(keypoint, "score", pose_score))

                keypoints[index] = [
                    np.clip(x, 0, frame_width - 1),
                    np.clip(y, 0, frame_height - 1),
                    score,
                ]

            bbox = bbox_from_keypoints(keypoints, frame_width, frame_height)
            people.append(Person(keypoints=keypoints, score=pose_score, bbox=bbox))

        return people

    def _label_to_index(self, label: Any) -> int | None:
        if hasattr(label, "name"):
            name = label.name
        else:
            name = str(label)

        name = name.lower()
        name = name.split(".")[-1]
        name = name.replace("_", " ")
        name = name.replace("-", " ")

        return CORAL_KEYPOINT_TO_INDEX.get(name)

    def _point_to_xy(self, point: Any) -> Tuple[float, float]:
        if hasattr(point, "x") and hasattr(point, "y"):
            return float(point.x), float(point.y)

        if isinstance(point, dict):
            return float(point["x"]), float(point["y"])

        return float(point[0]), float(point[1])

    def _map_point_to_frame(
        self,
        raw_x: float,
        raw_y: float,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float]:
        # Some wrappers return normalized coordinates.
        if 0.0 <= raw_x <= 1.5 and 0.0 <= raw_y <= 1.5:
            return raw_x * frame_width, raw_y * frame_height

        # Google Coral PoseEngine normally returns model-input pixel coordinates.
        x = raw_x * frame_width / max(1, self.input_width)
        y = raw_y * frame_height / max(1, self.input_height)

        return x, y