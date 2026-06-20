from __future__ import annotations
from iot_server import start_iot_server, update_iot_state

"""
Raspberry Pi 4 + Coral USB TPU fall detector using Coral PoseNet.

Purpose:
- Webcam at 640x480.
- Coral PoseNet returns multiple people and 17 keypoints per person.
- If more than one accepted person is visible, fall detection is disabled.
- If exactly one accepted person is visible, event-based fall logic runs.
- No argparse. Change settings below.

Expected folder layout on the Pi:
fall-detection/
  pi4_coral_posenet_fall.py
  fall_core.py
  project-posenet/
    pose_engine.py
    posenet_lib/...
    models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite

Recommended setup:
cd ~/fall-detection
git clone https://github.com/google-coral/project-posenet.git
cd project-posenet
sh install_requirements.sh
cd ..
python3 pi4_coral_posenet_fall.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .fall_core import (
    FallConfig,
    FallDetector,
    Pose,
    PersonState,
    SKELETON_EDGES,
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
PROJECT_POSENET_DIR = Path("project-posenet")
MODEL_PATH = PROJECT_POSENET_DIR / "models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite"

CONFIG = FallConfig(
    min_pose_score=0.10,
    min_kpt_score=0.08,
    min_valid_keypoints=5,
    min_body_area=0.015,

    stop_when_multiple_people=True,
    multi_person_confirm_frames=2,

    upright_angle=65.0,
    upright_max_ratio=1.00,

    horizontal_angle=35.0,
    horizontal_ratio=1.30,

    low_horizontal_angle=35.0,
    low_horizontal_ratio=0.95,

    pair_horizontal_ratio=1.55,
    pair_threshold_y=0.12,
    pair_threshold_x=0.25,

    fall_drop_speed=0.95,
    soft_drop_speed=0.35,
    motion_memory_sec=1.75,
    descent_timeout_sec=2.25,
    upright_memory_sec=5.00,

    min_low_drop_norm=0.08,
    min_low_drop_body_heights=0.25,

    fall_frames=3,
    high_confidence_increment=2,
    alarm_hold_sec=5.0,

    allow_static_lying=False,
    allow_no_upright_if_very_fast=True,
    very_fast_drop_speed=2.50,

    bed_top_y=None,
)

# Public boolean for IoT logic.
fall_detected = False
fall_alarm_until = 0.0

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
        if not project_dir.exists():
            raise FileNotFoundError(
                f"Missing {project_dir}. Clone it with:\n"
                "git clone https://github.com/google-coral/project-posenet.git"
            )
        if not model_path.exists():
            raise FileNotFoundError(f"Missing PoseNet model: {model_path}")

        sys.path.insert(0, str(project_dir.resolve()))
        from pose_engine import PoseEngine  # type: ignore

        self.engine = PoseEngine(str(model_path))
        shape = self.engine.get_input_tensor_shape()
        self.input_height = int(shape[1])
        self.input_width = int(shape[2])

        print("[MODEL] Coral PoseNet loaded:", model_path)
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


def draw_pose(frame: np.ndarray, pose: Pose, state: Optional[PersonState], disabled: bool = False) -> None:
    h, w = frame.shape[:2]
    keypoints = pose.keypoints
    min_kpt_score = CONFIG.min_kpt_score

    if disabled:
        color = (0, 255, 255)
    elif state is not None and state.last_status == "FALL":
        color = (0, 0, 255)
    elif state is not None and state.last_status == "LYING":
        color = (255, 180, 0)
    elif state is not None and state.last_status in {"POSSIBLE_FALL", "DESCENDING"}:
        color = (0, 165, 255)
    elif state is not None and state.last_status == "BENDING":
        color = (255, 255, 0)
    else:
        color = (0, 200, 0)

    for a, b in SKELETON_EDGES:
        ya, xa, sa = keypoints[a]
        yb, xb, sb = keypoints[b]
        if sa >= min_kpt_score and sb >= min_kpt_score:
            cv2.line(frame, (int(xa * w), int(ya * h)), (int(xb * w), int(yb * h)), color, 2)

    for y, x, score in keypoints:
        if score >= min_kpt_score:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 255), -1)

    bbox = pose_bbox_from_keypoints(keypoints, min_kpt_score) or pose.bbox
    ymin, xmin, ymax, xmax = bbox
    cv2.rectangle(frame, (int(xmin * w), int(ymin * h)), (int(xmax * w), int(ymax * h)), color, 2)

    if disabled:
        label = "MULTI-PERSON: FALL OFF"
    elif state is None:
        label = f"score={pose.score:.2f}"
    else:
        dbg = state.debug
        label = (
            f"id={state.track_id} {state.last_status} cnt={dbg.get('fall_counter', 0)} "
            f"r={float(dbg.get('ratio', 0)):.2f} "
            f"a={float(dbg.get('angle', -1)):.0f} "
            f"v={float(dbg.get('max_down_speed', 0)):.2f}"
        )
    cv2.putText(frame, label, (int(xmin * w), max(20, int(ymin * h) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_hud(
    frame: np.ndarray,
    current_fall_detected: bool,
    fps: float,
    people_count: int,
    model_info: Dict[str, object],
    disabled_reason: Optional[str],
) -> None:
    if disabled_reason:
        status_text = f"fall_detected = False | {disabled_reason}"
        status_color = (0, 255, 255)
    else:
        status_text = f"fall_detected = {current_fall_detected}"
        status_color = (0, 0, 255) if current_fall_detected else (0, 200, 0)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_color, 2)
    cv2.putText(frame, f"FPS(avg): {fps:.1f} | accepted people: {people_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(frame, f"model: {model_info.get('model_type', '?')} | infer: {float(model_info.get('inference_ms', 0.0)):.1f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)


def main() -> int:

    start_iot_server(port=8000)
    global fall_detected, fall_alarm_until

    model = CoralPoseNet(MODEL_PATH, PROJECT_POSENET_DIR)
    detector = FallDetector(CONFIG)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {CAMERA_INDEX}")

    frame_idx = 0
    processed_frames = 0
    start_time = time.time()
    multi_person_hits = 0

    print("[INFO] Started Coral PoseNet fall detector.")
    print("[INFO] Camera: 640x480 webcam")
    print("[INFO] Rule: if 2+ people are visible, fall detection is OFF.")
    print("[INFO] Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[INFO] Camera read failed.")
                break

            frame_idx += 1
            if MIRROR_IMAGE:
                frame = cv2.flip(frame, 1)

            now = time.time()
            poses, model_info = model.infer(frame)
            accepted = detector.accepted_poses(poses)
            people_count = len(accepted)

            if people_count > 1:
                multi_person_hits += 1
            else:
                multi_person_hits = 0

            multi_person_disabled = CONFIG.stop_when_multiple_people and multi_person_hits >= CONFIG.multi_person_confirm_frames
            disabled_reason = None
            results: List[Tuple[Pose, PersonState]] = []

            if people_count == 0:
                detector.reset()
                fall_alarm_until = 0.0
                fall_detected = False
                disabled_reason = "NO PERSON"
            elif multi_person_disabled:
                detector.reset()
                fall_alarm_until = 0.0
                fall_detected = False
                disabled_reason = "MULTI-PERSON: DETECTION STOPPED"
            else:
                results = detector.update(accepted[:1], now)
                current_fall = any(state.last_status == "FALL" for state in detector.states.values())
                if current_fall:
                    fall_alarm_until = now + CONFIG.alarm_hold_sec
                fall_detected = now <= fall_alarm_until

            processed_frames += 1
            elapsed = max(1e-6, time.time() - start_time)
            fps = processed_frames / elapsed

            draw_hud(frame, fall_detected, fps, people_count, model_info, disabled_reason)

            # Send latest frame + fall status to the IoT server.
            # The APK reads this through /video_feed, /status, and /ws.
            update_iot_state(
                frame,
                fall_detected=fall_detected,
                status=disabled_reason if disabled_reason else ("FALL" if fall_detected else "OK"),
                people=people_count,
                fps=fps,
            )

            if DEBUG_EVERY_N_FRAMES and frame_idx % DEBUG_EVERY_N_FRAMES == 0:
                debug_states = [st.debug for _, st in results]
                print(
                    f"[DEBUG] frame={frame_idx} fps={fps:.1f} people={people_count} "
                    f"fall_detected={fall_detected} states={debug_states}"
                )

            if DISPLAY:
                cv2.imshow("Pi4 Coral PoseNet Fall Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if DISPLAY:
            cv2.destroyAllWindows()

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
