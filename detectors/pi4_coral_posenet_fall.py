from __future__ import annotations
from iot_server import start_iot_server, update_iot_state
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from app_common import AppOptions, run_app
from .fall_core import (
    FallConfig,
    Pose,
    pose_bbox_from_keypoints,
)

# ============================================================
# User settings. Edit these, do not use command-line arguments.
# ============================================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
DISPLAY = True
MIRROR_IMAGE = False
DEBUG_EVERY_N_FRAMES = 30
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_POSENET_DIR = PROJECT_ROOT / "project-posenet"
MODEL_PATH = PROJECT_ROOT / "models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite"

CONFIG = FallConfig(
    # Pose quality: Coral PoseNet is noisier than MoveNet.
    min_pose_score=0.08,
    min_kpt_score=0.05,
    min_valid_keypoints=4,
    min_body_area=0.008,

    # Multiple people.
    stop_when_multiple_people=True,
    multi_person_confirm_frames=3,

    # Upright detection.
    upright_angle=65.0,
    upright_max_ratio=1.10,

    # Normal horizontal fall rule.
    horizontal_angle=55.0,
    horizontal_ratio=1.15,

    # Low / compact fall rule.
    low_horizontal_angle=65.0,
    low_horizontal_ratio=0.75,

    # Upper-body / lower-body spread rule.
    pair_horizontal_ratio=1.25,
    pair_threshold_y=0.18,
    pair_threshold_x=0.15,

    # Motion.
    fall_drop_speed=0.50,
    soft_drop_speed=0.18,
    motion_memory_sec=2.50,
    descent_timeout_sec=3.00,
    upright_memory_sec=6.00,

    # Low body check.
    min_low_drop_norm=0.025,
    min_low_drop_body_heights=0.08,

    # Confirmation.
    fall_frames=2,
    high_confidence_increment=2,
    alarm_hold_sec=5.0,

    # Safety rules.
    allow_static_lying=False,
    allow_no_upright_if_very_fast=True,
    very_fast_drop_speed=1.60,

    bed_top_y=None,
)
# PoseNet keypoint names in google-coral/project-posenet.
POSENET_NAME_TO_INDEX = {
    "NOSE": 0,
    "LEFT_EYE": 1,
    "RIGHT_EYE": 2,
    "LEFT_EAR": 3,
    "RIGHT_EAR": 4,
    "LEFT_SHOULDER": 5,
    "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW": 7,
    "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9,
    "RIGHT_WRIST": 10,
    "LEFT_HIP": 11,
    "RIGHT_HIP": 12,
    "LEFT_KNEE": 13,
    "RIGHT_KNEE": 14,
    "LEFT_ANKLE": 15,
    "RIGHT_ANKLE": 16,
}


class CoralPoseNet:
    def __init__(self, model_path: Path, project_dir: Path) -> None:
        model_path = Path(model_path).resolve()
        project_dir = Path(project_dir).resolve()

        if not project_dir.exists():
            raise FileNotFoundError(
                f"Missing {project_dir}. Clone it with:\n"
                "git clone https://github.com/google-coral/project-posenet.git"
            )

        if not model_path.exists():
            raise FileNotFoundError(f"Missing PoseNet model: {model_path}")

        pose_engine_path = project_dir / "pose_engine.py"
        if not pose_engine_path.exists():
            raise FileNotFoundError(f"Missing pose_engine.py: {pose_engine_path}")

        decoder_path = (
            project_dir
            / "posenet_lib"
            / os.uname().machine
            / "posenet_decoder.so"
        )
        if not decoder_path.exists():
            raise FileNotFoundError(f"Missing PoseNet decoder: {decoder_path}")

        if str(project_dir) not in sys.path:
            sys.path.insert(0, str(project_dir))

        old_cwd = os.getcwd()
        try:
            os.chdir(str(project_dir))
            from pose_engine import PoseEngine  # type: ignore

            self.engine = PoseEngine(str(model_path))
        finally:
            os.chdir(old_cwd)

        shape = self.engine.get_input_tensor_shape()
        self.input_height = int(shape[1])
        self.input_width = int(shape[2])

        print("[MODEL] Coral PoseNet loaded:", model_path)
        print("[MODEL] project-posenet:", project_dir)
        print("[MODEL] decoder:", decoder_path)
        print("[MODEL] input shape:", list(shape))

    def infer(self, frame_bgr: np.ndarray) -> Tuple[List[Pose], Dict[str, object]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        coral_poses, inference_time = self.engine.DetectPosesInImage(pil_img)
        poses = [self._convert_pose(p) for p in coral_poses]
        return poses, {
            "model_type": "coral_posenet",
            "inference_ms": float(inference_time),
            "raw_pose_scores": [float(p.score) for p in poses],
        }

    def _convert_pose(self, coral_pose: object) -> Pose:
        keypoints = np.zeros((17, 3), dtype=float)
        keypoints[:, 0] = 0.5
        keypoints[:, 1] = 0.5

        for label, kp in coral_pose.keypoints.items():
            name = getattr(label, "name", str(label)).split(".")[-1]
            idx = POSENET_NAME_TO_INDEX.get(name)
            if idx is None:
                continue

            # PoseEngine gives x/y in model input coordinates.
            x_norm = float(np.clip(kp.point.x / max(self.input_width, 1), 0.0, 1.0))
            y_norm = float(np.clip(kp.point.y / max(self.input_height, 1), 0.0, 1.0))
            keypoints[idx] = [y_norm, x_norm, float(kp.score)]

        bbox = pose_bbox_from_keypoints(keypoints, min_score=0.01)
        if bbox is None:
            bbox = (0.0, 0.0, 1.0, 1.0)

        return Pose(keypoints=keypoints, bbox=bbox, score=float(coral_pose.score))


CoralPoseNetDetector = CoralPoseNet


def main() -> int:
    model = CoralPoseNet(MODEL_PATH, PROJECT_POSENET_DIR)
    start_iot_server(port=8000)
    options = AppOptions(
        title="Pi4 Coral PoseNet Fall Detection",
        camera_index=CAMERA_INDEX,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        camera_fps=CAMERA_FPS,
        display=DISPLAY,
        draw_pose=False,
        mirror_image=MIRROR_IMAGE,
        debug_every_n_frames=DEBUG_EVERY_N_FRAMES,
    )
    return run_app(model, CONFIG, options, state_sink=update_iot_state)


if __name__ == "__main__":
    raise SystemExit(main())
