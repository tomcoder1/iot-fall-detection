from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

DEFAULT_MODEL_PATH = "movenet_multipose_lightning.tflite"
DEFAULT_VIDEO_SOURCE = "0"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Pi 4: 1 or 2 is usually safer. Laptop: 2 to 4 is fine.
NUM_THREADS = 2

MIN_POSE_SCORE = 0.10
MIN_KPT_SCORE = 0.08

ANGLE_THRESHOLD = 60.0          # larger = easier to call horizontal
RATIO_THRESHOLD = 0.85          # smaller = easier to call lying/horizontal
HIP_DROP_SPEED = 0.35           # Used for debug/ordinary downward movement.
SHOULDER_DROP_SPEED = 0.35

# A real fall needs stronger transition motion than ordinary slow lying or pose jitter.
# Increase this if slow lying still becomes FALL. Lower it if real falls only show LYING.
FALL_DROP_SPEED = 1.00
FALL_MOTION_MEMORY_SEC = 0.90

PAIR_THRESHOLD_Y = 0.12
PAIR_THRESHOLD_X = 0.25

FALL_FRAMES = 2
MOTION_MEMORY_SEC = 2.0
ALARM_HOLD_SEC = 5.0

# Keep False for real fall detection. Slow intentional lying becomes LYING, not FALL.
# Set True only if you intentionally want static lying to be treated more aggressively.
ALLOW_STATIC_LYING = False

# Optional bed/sofa cancellation line. Example: 0.55 means if upper body is
# below 55% of image height, cancel the fall. Keep None unless you calibrate it.
BED_TOP_Y: Optional[float] = None

# If > 0, process only every N+1 frame. Example: 1 means process every other frame.
SKIP_FRAMES = 0

# If more than one accepted person appears, fall detection stops.
STOP_WHEN_MULTIPLE_PEOPLE = True

# Dynamic model input fallback sizes
MULTIPOSE_INPUT_SIZE = 256
SINGLEPOSE_INPUT_SIZE = 192

# Public boolean for IoT logic
fall_detected = False
fall_alarm_until = 0.0

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

UPPER_BODY = [
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
]

LOWER_BODY = [
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]

SKELETON_EDGES = [
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
]

@dataclass
class Pose:
    keypoints: np.ndarray  # shape: [17, 3], normalized y, x, score
    bbox: Tuple[float, float, float, float]  # ymin, xmin, ymax, xmax normalized
    score: float


@dataclass
class PersonState:
    track_id: int
    center: Tuple[float, float]
    last_seen: float
    prev_hip_y: Optional[float] = None
    prev_shoulder_y: Optional[float] = None
    prev_time: Optional[float] = None
    fall_counter: int = 0
    recent_motion_until: float = 0.0
    was_lying: bool = False
    last_status: str = "UNKNOWN"
    debug: Dict[str, Union[float, bool, int, str]] = field(default_factory=dict)

def source_from_string(value: str) -> Union[int, str]:
    value = str(value).strip()
    if value.isdigit():
        return int(value)
    return value

def safe_int_list(values: Iterable[object]) -> List[int]:
    return [int(v) for v in values]

def letterbox_bgr_to_rgb(
    frame_bgr: np.ndarray,
    target_w: int,
    target_h: int,
) -> Tuple[np.ndarray, Dict[str, float]]:

    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid frame size.")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    meta = {
        "orig_w": float(w),
        "orig_h": float(h),
        "target_w": float(target_w),
        "target_h": float(target_h),
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
    }
    return rgb, meta


def input_norm_to_orig_norm(y: float, x: float, meta: Dict[str, float]) -> Tuple[float, float]:
    target_w = meta["target_w"]
    target_h = meta["target_h"]
    scale = meta["scale"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]

    x_input = x * target_w
    y_input = y * target_h

    x_orig = (x_input - pad_x) / max(scale, 1e-6)
    y_orig = (y_input - pad_y) / max(scale, 1e-6)

    x_norm = float(np.clip(x_orig / max(orig_w, 1e-6), 0.0, 1.0))
    y_norm = float(np.clip(y_orig / max(orig_h, 1e-6), 0.0, 1.0))

    return y_norm, x_norm

class TFLiteMoveNet:
    def __init__(
        self,
        model_path: Union[str, Path],
        num_threads: int = 2,
        force_input_size: Optional[int] = None,
        debug: bool = True,
    ):
        self.model_path = str(model_path)
        self.debug = debug

        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
            runtime_name = "tflite_runtime"
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
                runtime_name = "tensorflow.lite"
            except ImportError as exc:
                raise RuntimeError() from exc

        self.interpreter = Interpreter(model_path=self.model_path, num_threads=num_threads)

        pre_input_details = self.interpreter.get_input_details()
        if not pre_input_details:
            raise RuntimeError("Model has no input tensor.")

        pre_input = pre_input_details[0]
        pre_shape = safe_int_list(pre_input["shape"])
        input_index = int(pre_input["index"])

        target_size = force_input_size
        if target_size is None:
            lower_name = Path(self.model_path).name.lower()
            if "multi" in lower_name:
                target_size = MULTIPOSE_INPUT_SIZE
            elif "single" in lower_name or "movenet" in lower_name:
                target_size = SINGLEPOSE_INPUT_SIZE
            else:
                target_size = MULTIPOSE_INPUT_SIZE

        resized_dynamic = False
        if len(pre_shape) == 4 and (pre_shape[1] <= 1 or pre_shape[2] <= 1):
            self.interpreter.resize_tensor_input(
                input_index,
                [1, int(target_size), int(target_size), 3],
                strict=False,
            )
            resized_dynamic = True
        elif len(pre_shape) == 4 and force_input_size is not None:
            self.interpreter.resize_tensor_input(
                input_index,
                [1, int(force_input_size), int(force_input_size), 3],
                strict=False,
            )
            resized_dynamic = True

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        if not self.output_details:
            raise RuntimeError("Model has no output tensor.")

        self.input_index = int(self.input_details[0]["index"])
        self.input_shape = safe_int_list(self.input_details[0]["shape"])
        self.input_dtype = self.input_details[0]["dtype"]
        self.input_quantization = self.input_details[0].get("quantization", (0.0, 0))

        if len(self.input_shape) != 4:
            raise RuntimeError(f"Expected 4D input tensor, got shape {self.input_shape}")

        self.input_height = int(self.input_shape[1])
        self.input_width = int(self.input_shape[2])

        if self.input_height <= 1 or self.input_width <= 1:
            raise RuntimeError(
                f"Model input is still invalid after resize: {self.input_shape}. "
                "Expected something like [1, 256, 256, 3]."
            )

        if self.debug:
            print("[MODEL] runtime:", runtime_name)
            print("[MODEL] path:", self.model_path)
            print("[MODEL] original input shape:", pre_shape)
            print("[MODEL] resized dynamic input:", resized_dynamic)
            print("[MODEL] active input shape:", self.input_shape)
            print("[MODEL] input dtype:", self.input_dtype)
            print("[MODEL] input quantization:", self.input_quantization)
            for i, output_detail in enumerate(self.output_details):
                print(
                    f"[MODEL] output {i} shape:",
                    output_detail["shape"],
                    "dtype:",
                    output_detail["dtype"],
                )

    def _prepare_tensor(self, input_rgb: np.ndarray) -> np.ndarray:
        tensor = input_rgb[np.newaxis, ...]

        if self.input_dtype == np.float32:
            return tensor.astype(np.float32) / 255.0

        if self.input_dtype == np.uint8:
            return tensor.astype(np.uint8)

        if self.input_dtype == np.int8:
            scale, zero_point = self.input_quantization
            if scale and scale > 0:
                float_tensor = tensor.astype(np.float32) / 255.0
                quantized = float_tensor / float(scale) + int(zero_point)
                return np.clip(np.round(quantized), -128, 127).astype(np.int8)
            return tensor.astype(np.int8)

        if self.input_dtype == np.int32:
            return tensor.astype(np.int32)

        return tensor.astype(self.input_dtype)

    def infer(self, frame_bgr: np.ndarray) -> Tuple[List[Pose], Dict[str, object]]:
        input_rgb, meta = letterbox_bgr_to_rgb(frame_bgr, self.input_width, self.input_height)
        tensor = self._prepare_tensor(input_rgb)

        self.interpreter.set_tensor(self.input_index, tensor)
        self.interpreter.invoke()

        outputs = [self.interpreter.get_tensor(d["index"]) for d in self.output_details]
        poses, info = parse_movenet_outputs(outputs, meta)
        return poses, info


def parse_movenet_outputs(
    outputs: List[np.ndarray],
    meta: Dict[str, float],
) -> Tuple[List[Pose], Dict[str, object]]:
    """Parse MultiPose or SinglePose MoveNet outputs."""
    if not outputs:
        return [], {"model_type": "unknown", "raw_pose_scores": []}

    output = np.asarray(outputs[0])
    arr = np.squeeze(output)

    # MultiPose: [1, 6, 56] -> [6, 56]
    if arr.ndim == 1 and arr.size == 56:
        arr = arr.reshape(1, 56)

    if arr.ndim == 2 and arr.shape[-1] == 56:
        poses = parse_multipose_array(arr, meta)
        scores = [float(row[55]) for row in arr]
        return poses, {"model_type": "multipose", "raw_pose_scores": scores}

    # SinglePose common shapes:
    # [1, 1, 17, 3] squeezed -> [17, 3]
    # [1, 17, 3] squeezed -> [17, 3]
    if arr.ndim == 2 and arr.shape == (17, 3):
        pose = parse_singlepose_array(arr, meta)
        score = pose.score if pose is not None else 0.0
        return ([pose] if pose is not None else []), {
            "model_type": "singlepose",
            "raw_pose_scores": [float(score)],
        }

    # Some models may return [1, 17, 3] without squeezing to exactly [17,3]
    if arr.ndim == 3 and arr.shape[-2:] == (17, 3):
        one = arr[0]
        pose = parse_singlepose_array(one, meta)
        score = pose.score if pose is not None else 0.0
        return ([pose] if pose is not None else []), {
            "model_type": "singlepose",
            "raw_pose_scores": [float(score)],
        }

    raise RuntimeError(
        f"Unexpected MoveNet output shape. Raw shape={output.shape}, squeezed shape={arr.shape}. "
        "Expected MultiPose [1,6,56] or SinglePose [1,1,17,3]."
    )


def parse_multipose_array(arr: np.ndarray, meta: Dict[str, float]) -> List[Pose]:
    poses: List[Pose] = []

    for row in arr:
        kpts = row[:51].reshape(17, 3).astype(float)
        bbox_raw = row[51:55].astype(float)
        pose_score = float(row[55])

        converted = np.zeros_like(kpts, dtype=float)
        for i, (y, x, score) in enumerate(kpts):
            yy, xx = input_norm_to_orig_norm(float(y), float(x), meta)
            converted[i] = [yy, xx, float(score)]

        ymin, xmin = input_norm_to_orig_norm(float(bbox_raw[0]), float(bbox_raw[1]), meta)
        ymax, xmax = input_norm_to_orig_norm(float(bbox_raw[2]), float(bbox_raw[3]), meta)

        bbox = (
            min(ymin, ymax),
            min(xmin, xmax),
            max(ymin, ymax),
            max(xmin, xmax),
        )

        poses.append(Pose(keypoints=converted, bbox=bbox, score=pose_score))

    return poses


def parse_singlepose_array(arr: np.ndarray, meta: Dict[str, float]) -> Optional[Pose]:
    kpts = arr.astype(float)
    if kpts.shape != (17, 3):
        return None

    converted = np.zeros_like(kpts, dtype=float)
    scores: List[float] = []

    for i, (y, x, score) in enumerate(kpts):
        yy, xx = input_norm_to_orig_norm(float(y), float(x), meta)
        converted[i] = [yy, xx, float(score)]
        scores.append(float(score))

    bbox = pose_bbox_from_keypoints(converted, min_score=0.01)
    if bbox is None:
        bbox = (0.0, 0.0, 1.0, 1.0)

    # SinglePose does not provide person score directly. Use average of useful body keypoints.
    body_scores = [converted[i, 2] for i in [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]]
    pose_score = float(np.mean(body_scores))

    return Pose(keypoints=converted, bbox=bbox, score=pose_score)


# ============================================================
# Pose geometry helpers
# ============================================================

def valid_points(keypoints: np.ndarray, indices: Iterable[int], min_score: float) -> np.ndarray:
    points = []
    for idx in indices:
        y, x, score = keypoints[idx]
        if score >= min_score:
            points.append([float(y), float(x), float(score)])

    if not points:
        return np.empty((0, 3), dtype=float)

    return np.asarray(points, dtype=float)


def midpoint(
    keypoints: np.ndarray,
    left_idx: int,
    right_idx: int,
    min_score: float,
) -> Optional[Tuple[float, float]]:
    points = []

    for idx in (left_idx, right_idx):
        y, x, score = keypoints[idx]
        if score >= min_score:
            points.append((float(y), float(x)))

    if not points:
        return None

    return float(np.mean([p[0] for p in points])), float(np.mean([p[1] for p in points]))


def pose_bbox_from_keypoints(
    keypoints: np.ndarray,
    min_score: float,
) -> Optional[Tuple[float, float, float, float]]:
    points = valid_points(keypoints, range(17), min_score)

    if len(points) < 4:
        return None

    ymin = float(np.min(points[:, 0]))
    ymax = float(np.max(points[:, 0]))
    xmin = float(np.min(points[:, 1]))
    xmax = float(np.max(points[:, 1]))

    return ymin, xmin, ymax, xmax


def bbox_ratio(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    ymin, xmin, ymax, xmax = bbox

    height = max(1e-6, ymax - ymin)
    width = max(1e-6, xmax - xmin)

    return width / height, width, height


def torso_angle_degrees(keypoints: np.ndarray, min_score: float) -> Optional[float]:
    """
    Returns torso angle in degrees, where:
    - near 90 degrees means vertical/upright
    - near 0 degrees means horizontal/lying
    """
    shoulder = midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, min_score)
    hip = midpoint(keypoints, LEFT_HIP, RIGHT_HIP, min_score)

    if shoulder is None or hip is None:
        return None

    dy = abs(shoulder[0] - hip[0])
    dx = abs(shoulder[1] - hip[1])

    return float(math.degrees(math.atan2(dy, dx + 1e-6)))


def pair_rule_upper_lower(
    keypoints: np.ndarray,
    min_score: float,
    threshold_y: float,
    threshold_x: float,
) -> bool:
    """
    Checks whether an upper-body point and lower-body point are almost level
    vertically but far apart horizontally, which often happens when lying down.
    """
    upper = valid_points(keypoints, UPPER_BODY, min_score)
    lower = valid_points(keypoints, LOWER_BODY, min_score)

    if len(upper) == 0 or len(lower) == 0:
        return False

    for upper_y, upper_x, _ in upper:
        for lower_y, lower_x, _ in lower:
            y_close = abs(float(upper_y) - float(lower_y)) <= threshold_y
            x_far = abs(float(upper_x) - float(lower_x)) >= threshold_x

            if y_close and x_far:
                return True

    return False


def is_on_bed_region(
    keypoints: np.ndarray,
    min_score: float,
    bed_top_y: Optional[float],
) -> bool:
    if bed_top_y is None:
        return False

    upper = valid_points(keypoints, UPPER_BODY, min_score)

    if len(upper) == 0:
        return False

    upper_y = float(np.median(upper[:, 0]))

    return upper_y > bed_top_y


def center_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    ymin, xmin, ymax, xmax = bbox
    return float((ymin + ymax) / 2.0), float((xmin + xmax) / 2.0)


# ============================================================
# Fall detector
# ============================================================

class FallDetector:
    def __init__(
        self,
        min_pose_score: float,
        min_kpt_score: float,
        angle_threshold: float,
        ratio_threshold: float,
        hip_drop_speed: float,
        shoulder_drop_speed: float,
        pair_threshold_y: float,
        pair_threshold_x: float,
        fall_frames: int,
        motion_memory_sec: float,
        bed_top_y: Optional[float],
        allow_static_lying: bool,
        track_distance: float = 0.30,
        debug_rules: bool = False,
    ):
        self.min_pose_score = min_pose_score
        self.min_kpt_score = min_kpt_score
        self.angle_threshold = angle_threshold
        self.ratio_threshold = ratio_threshold
        self.hip_drop_speed = hip_drop_speed
        self.shoulder_drop_speed = shoulder_drop_speed
        self.pair_threshold_y = pair_threshold_y
        self.pair_threshold_x = pair_threshold_x
        self.fall_frames = fall_frames
        self.motion_memory_sec = motion_memory_sec
        self.bed_top_y = bed_top_y
        self.allow_static_lying = allow_static_lying
        self.track_distance = track_distance
        self.debug_rules = debug_rules

        self.states: Dict[int, PersonState] = {}
        self.next_track_id = 1

    def reset(self) -> None:
        self.states.clear()
        self.next_track_id = 1

    def accepted_poses(self, poses: List[Pose]) -> List[Pose]:
        candidates = [pose for pose in poses if pose.score >= self.min_pose_score]
        candidates.sort(key=lambda pose: pose.score, reverse=True)
        return candidates

    def update(self, poses: List[Pose], now: float) -> List[Tuple[Pose, PersonState]]:
        results: List[Tuple[Pose, PersonState]] = []

        candidates = self.accepted_poses(poses)
        assigned_tracks = set()

        for pose in candidates:
            state = self._assign_track(pose, now, assigned_tracks, len(candidates))
            assigned_tracks.add(state.track_id)

            self._update_state_with_pose(state, pose, now)
            results.append((pose, state))

        stale_tracks = [
            track_id
            for track_id, state in self.states.items()
            if now - state.last_seen > 3.0
        ]

        for track_id in stale_tracks:
            del self.states[track_id]

        return results

    def _assign_track(
        self,
        pose: Pose,
        now: float,
        assigned_tracks: set,
        candidate_count: int,
    ) -> PersonState:
        bbox = pose_bbox_from_keypoints(pose.keypoints, self.min_kpt_score) or pose.bbox
        center = center_from_bbox(bbox)

        # In single-person mode, keep the existing track even if the center jumps.
        # This avoids losing velocity history during a fast fall.
        if candidate_count == 1 and len(self.states) == 1 and len(assigned_tracks) == 0:
            state = next(iter(self.states.values()))
            state.center = center
            state.last_seen = now
            return state

        best_id = None
        best_dist = 1e9

        for track_id, state in self.states.items():
            if track_id in assigned_tracks:
                continue

            dist = math.hypot(center[0] - state.center[0], center[1] - state.center[1])

            if dist < best_dist:
                best_dist = dist
                best_id = track_id

        if best_id is not None and best_dist <= self.track_distance:
            state = self.states[best_id]
            state.center = center
            state.last_seen = now
            return state

        track_id = self.next_track_id
        self.next_track_id += 1

        state = PersonState(track_id=track_id, center=center, last_seen=now)
        self.states[track_id] = state

        return state

    def _update_state_with_pose(self, state: PersonState, pose: Pose, now: float) -> None:
        keypoints = pose.keypoints

        bbox = pose_bbox_from_keypoints(keypoints, self.min_kpt_score) or pose.bbox
        ratio, body_w, body_h = bbox_ratio(bbox)

        angle = torso_angle_degrees(keypoints, self.min_kpt_score)
        angle_rule = angle is not None and angle < self.angle_threshold
        ratio_rule = ratio > self.ratio_threshold

        pair_rule = pair_rule_upper_lower(
            keypoints,
            self.min_kpt_score,
            self.pair_threshold_y,
            self.pair_threshold_x,
        )

        bed_rule = is_on_bed_region(keypoints, self.min_kpt_score, self.bed_top_y)

        hip = midpoint(keypoints, LEFT_HIP, RIGHT_HIP, self.min_kpt_score)
        shoulder = midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, self.min_kpt_score)

        hip_speed = 0.0
        shoulder_speed = 0.0
        dt = None

        if state.prev_time is not None:
            dt = max(1e-3, now - state.prev_time)

        if hip is not None and state.prev_hip_y is not None and dt is not None:
            hip_speed = ((hip[0] - state.prev_hip_y) / dt) / max(body_h, 1e-6)

        if shoulder is not None and state.prev_shoulder_y is not None and dt is not None:
            shoulder_speed = ((shoulder[0] - state.prev_shoulder_y) / dt) / max(body_h, 1e-6)

        prev_was_lying = state.was_lying

        # Ordinary downward movement is kept for debug.
        # It is NOT enough to create a fall event.
        sudden_motion = (
            hip_speed > self.hip_drop_speed
            or shoulder_speed > self.shoulder_drop_speed
        )

        # A real fall event needs a stronger downward transition while the person
        # was not already lying. This prevents LYING from turning into FALL later
        # because of small pose jitter on the floor/bed.
        strong_fall_motion = (
            hip_speed > FALL_DROP_SPEED
            or shoulder_speed > FALL_DROP_SPEED
        )

        if strong_fall_motion and not prev_was_lying:
            state.recent_motion_until = now + FALL_MOTION_MEMORY_SEC

        recent_motion = now <= state.recent_motion_until

        # Body-state logic:
        # - LYING means the current pose looks horizontal/lying.
        # - FALL means a strong downward transition happened, then the body became lying.
        # - If the person is already lying, later jitter cannot upgrade LYING to FALL.
        geometry_fall = angle_rule and ratio_rule
        lying_candidate = geometry_fall or (ratio_rule and pair_rule)

        fall_candidate = (
            lying_candidate
            and recent_motion
            and (not prev_was_lying or state.fall_counter > 0)
        )

        if bed_rule:
            fall_candidate = False
            lying_candidate = False

        # Cap the counter so it does not grow endlessly while lying down.
        # This lets the state recover quickly after standing up.
        if fall_candidate:
            state.fall_counter = min(self.fall_frames, state.fall_counter + 1)
        else:
            state.fall_counter = max(0, state.fall_counter - 1)

        if state.fall_counter >= self.fall_frames:
            state.last_status = "FALL"
        elif state.fall_counter > 0:
            state.last_status = "POSSIBLE_FALL"
        elif lying_candidate:
            state.last_status = "LYING"
        else:
            state.last_status = "OK"

        state.was_lying = bool(lying_candidate)

        if hip is not None:
            state.prev_hip_y = hip[0]

        if shoulder is not None:
            state.prev_shoulder_y = shoulder[0]

        state.prev_time = now

        state.debug = {
            "pose_score": float(pose.score),
            "ratio": float(ratio),
            "angle": float(angle) if angle is not None else -1.0,
            "body_w": float(body_w),
            "body_h": float(body_h),
            "hip_speed": float(hip_speed),
            "shoulder_speed": float(shoulder_speed),
            "angle_rule": bool(angle_rule),
            "ratio_rule": bool(ratio_rule),
            "pair_rule": bool(pair_rule),
            "recent_motion": bool(recent_motion),
            "sudden_motion": bool(sudden_motion),
            "strong_fall_motion": bool(strong_fall_motion),
            "was_lying_before": bool(prev_was_lying),
            "lying_candidate": bool(lying_candidate),
            "bed_rule": bool(bed_rule),
            "fall_candidate": bool(fall_candidate),
            "fall_counter": int(state.fall_counter),
        }

        if self.debug_rules:
            print(
                f"[RULES] id={state.track_id} status={state.last_status} "
                f"cnt={state.fall_counter} ratio={ratio:.2f} "
                f"angle={(angle if angle is not None else -1):.1f} "
                f"hip_v={hip_speed:.2f} sho_v={shoulder_speed:.2f} "
                f"angle_rule={angle_rule} ratio_rule={ratio_rule} "
                f"pair_rule={pair_rule} recent_motion={recent_motion} "
                f"sudden={sudden_motion} strong={strong_fall_motion} "
                f"was_lying={prev_was_lying} lying={lying_candidate} "
                f"fall_candidate={fall_candidate}"
            )


# ============================================================
# Camera/video sources
# ============================================================

class FrameSource:
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    def timestamp(self, frame_idx: int) -> float:
        return time.time()


class OpenCVSource(FrameSource):
    def __init__(self, source: Union[int, str], width: int, height: int, fps: int):
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame

    def release(self) -> None:
        self.cap.release()

    def timestamp(self, frame_idx: int) -> float:
        # Webcam: use wall clock. Video file: use video timestamp/FPS.
        if isinstance(self.source, int):
            return time.time()

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 1e-3:
            return frame_idx / float(fps)

        msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        if msec > 0:
            return msec / 1000.0

        return time.time()


class Picamera2Source(FrameSource):
    def __init__(self, width: int, height: int, fps: int):
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is not installed. On Raspberry Pi OS, run:\n"
                "sudo apt install python3-picamera2\n"
                "Or use --camera-backend opencv for a USB webcam."
            ) from exc

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.5)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        frame_rgb = self.picam2.capture_array()
        if frame_rgb is None:
            return False, None
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return True, frame_bgr

    def release(self) -> None:
        try:
            self.picam2.stop()
        except Exception:
            pass


def open_frame_source(args: argparse.Namespace) -> FrameSource:
    backend = args.camera_backend.lower()
    source = source_from_string(args.source)

    if backend == "picamera2":
        return Picamera2Source(args.width, args.height, args.camera_fps)

    if backend == "opencv":
        return OpenCVSource(source, args.width, args.height, args.camera_fps)

    if backend == "auto":
        # Use OpenCV by default. It works for laptop and USB webcam on Pi.
        return OpenCVSource(source, args.width, args.height, args.camera_fps)

    raise ValueError(f"Unknown camera backend: {args.camera_backend}")


# ============================================================
# Drawing
# ============================================================

def draw_pose(
    frame: np.ndarray,
    pose: Pose,
    state: Optional[PersonState],
    min_kpt_score: float,
    disabled: bool = False,
) -> None:
    h, w = frame.shape[:2]
    keypoints = pose.keypoints

    if disabled:
        color = (0, 255, 255)
    elif state is not None and state.last_status == "FALL":
        color = (0, 0, 255)
    elif state is not None and state.last_status == "LYING":
        color = (255, 180, 0)
    elif state is not None and state.last_status == "POSSIBLE_FALL":
        color = (0, 165, 255)
    else:
        color = (0, 200, 0)

    for a, b in SKELETON_EDGES:
        ya, xa, score_a = keypoints[a]
        yb, xb, score_b = keypoints[b]

        if score_a >= min_kpt_score and score_b >= min_kpt_score:
            p1 = (int(xa * w), int(ya * h))
            p2 = (int(xb * w), int(yb * h))
            cv2.line(frame, p1, p2, color, 2)

    for y, x, score in keypoints:
        if score >= min_kpt_score:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 255), -1)

    bbox = pose_bbox_from_keypoints(keypoints, min_kpt_score) or pose.bbox
    ymin, xmin, ymax, xmax = bbox

    cv2.rectangle(
        frame,
        (int(xmin * w), int(ymin * h)),
        (int(xmax * w), int(ymax * h)),
        color,
        2,
    )

    if state is None:
        label = f"score={pose.score:.2f}"
    else:
        debug = state.debug
        max_speed = max(
            float(debug.get("hip_speed", 0.0)),
            float(debug.get("shoulder_speed", 0.0)),
        )

        label = (
            f"id={state.track_id} {state.last_status} "
            f"cnt={debug.get('fall_counter', 0)} "
            f"r={debug.get('ratio', 0):.2f} "
            f"a={debug.get('angle', -1):.0f} "
            f"v={max_speed:.2f}"
        )

    if disabled:
        label = "MULTI-PERSON: FALL OFF"

    text_x = int(xmin * w)
    text_y = max(20, int(ymin * h) - 8)

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def draw_hud(
    frame: np.ndarray,
    current_fall_detected: bool,
    fps: float,
    people_count: int,
    model_info: Dict[str, object],
    disabled_reason: Optional[str] = None,
) -> None:
    if disabled_reason:
        status_text = f"fall_detected = False | {disabled_reason}"
        status_color = (0, 255, 255)
    else:
        status_text = f"fall_detected = {current_fall_detected}"
        status_color = (0, 0, 255) if current_fall_detected else (0, 200, 0)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 102), (0, 0, 0), -1)

    cv2.putText(
        frame,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2,
    )

    cv2.putText(
        frame,
        f"FPS(avg): {fps:.1f} | accepted people: {people_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )

    scores = model_info.get("raw_pose_scores", [])
    if isinstance(scores, list):
        score_text = ",".join(f"{float(s):.2f}" for s in scores[:6])
    else:
        score_text = "?"

    cv2.putText(
        frame,
        f"model: {model_info.get('model_type', '?')} | raw scores: [{score_text}]",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
    )


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoveNet rule-based fall detector for laptop and Raspberry Pi 4.")

    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to .tflite MoveNet model.")
    parser.add_argument("--source", default=DEFAULT_VIDEO_SOURCE, help="Camera index like 0, or video file path.")
    parser.add_argument(
        "--camera-backend",
        default="auto",
        choices=["auto", "opencv", "picamera2"],
        help="Use opencv for laptop/USB webcam, picamera2 for Raspberry Pi CSI camera.",
    )
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT, help="Camera capture height.")
    parser.add_argument("--camera-fps", type=int, default=CAMERA_FPS, help="Requested camera FPS.")
    parser.add_argument("--threads", type=int, default=NUM_THREADS, help="TFLite interpreter thread count.")
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Force model input size. MultiPose usually 256. SinglePose usually 192.",
    )
    parser.add_argument("--skip-frames", type=int, default=SKIP_FRAMES, help="Skip frames for speed. 0 means no skip.")
    parser.add_argument("--no-display", action="store_true", help="Run without cv2.imshow window.")
    parser.add_argument("--debug", action="store_true", help="Print frame/model debug every 30 frames.")
    parser.add_argument("--debug-rules", action="store_true", help="Print fall rule values every processed frame.")
    parser.add_argument(
        "--allow-multiple-people",
        action="store_true",
        help="Do not disable fall detection when more than one person is visible. Not recommended.",
    )

    return parser.parse_args()


def main() -> int:
    global fall_detected, fall_alarm_until

    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Put the .tflite model in this folder or pass --model path/to/model.tflite"
        )

    movenet = TFLiteMoveNet(
        model_path=model_path,
        num_threads=args.threads,
        force_input_size=args.input_size,
        debug=True,
    )

    detector = FallDetector(
        min_pose_score=MIN_POSE_SCORE,
        min_kpt_score=MIN_KPT_SCORE,
        angle_threshold=ANGLE_THRESHOLD,
        ratio_threshold=RATIO_THRESHOLD,
        hip_drop_speed=HIP_DROP_SPEED,
        shoulder_drop_speed=SHOULDER_DROP_SPEED,
        pair_threshold_y=PAIR_THRESHOLD_Y,
        pair_threshold_x=PAIR_THRESHOLD_X,
        fall_frames=FALL_FRAMES,
        motion_memory_sec=MOTION_MEMORY_SEC,
        bed_top_y=BED_TOP_Y,
        allow_static_lying=ALLOW_STATIC_LYING,
        debug_rules=args.debug_rules,
    )

    source = open_frame_source(args)

    frame_idx = 0
    processed_frames = 0
    start_time = time.time()
    fps_display = 0.0

    print("[INFO] Started MoveNet fall detector.")
    print("[INFO] Press q to quit if display is enabled.")
    print("[INFO] Multi-person rule:", "OFF" if args.allow_multiple_people else "ON")
    print("[INFO] State logic: slow lying = LYING. Only strong transition + lying = FALL.")
    print("[INFO] Important: active input shape should be [1, 256, 256, 3] for MultiPose.")

    try:
        while True:
            ok, frame = source.read()

            if not ok or frame is None:
                print("[INFO] End of stream or camera read failed.")
                break

            frame_idx += 1

            if args.skip_frames > 0 and (frame_idx % (args.skip_frames + 1)) != 1:
                continue

            now = source.timestamp(frame_idx)

            poses, model_info = movenet.infer(frame)
            accepted = detector.accepted_poses(poses)
            accepted_people_count = len(accepted)

            multi_person_disabled = (
                not args.allow_multiple_people
                and STOP_WHEN_MULTIPLE_PEOPLE
                and accepted_people_count > 1
            )

            results: List[Tuple[Pose, PersonState]] = []
            disabled_reason = None

            if multi_person_disabled:
                # Safety behavior requested by user: more than 1 person means stop detecting.
                detector.reset()
                fall_alarm_until = 0.0
                fall_detected = False
                disabled_reason = "MULTI-PERSON: DETECTION STOPPED"
            else:
                results = detector.update(accepted, now)

                current_fall = any(
                    state.last_status == "FALL"
                    for state in detector.states.values()
                )

                if current_fall:
                    fall_alarm_until = now + ALARM_HOLD_SEC

                fall_detected = now <= fall_alarm_until

            processed_frames += 1
            elapsed = max(1e-6, time.time() - start_time)
            fps_display = processed_frames / elapsed

            if args.debug and frame_idx % 30 == 0:
                print(
                    f"[DEBUG] frame {frame_idx} shape={frame.shape} "
                    f"min={frame.min()} max={frame.max()} mean={float(frame.mean()):.2f}"
                )
                print("[DEBUG] model type:", model_info.get("model_type"))
                print("[DEBUG] raw pose scores:", [round(float(s), 3) for s in model_info.get("raw_pose_scores", [])])
                print("[DEBUG] accepted people:", accepted_people_count)
                print("[DEBUG] fall_detected:", fall_detected)

            if not args.no_display:
                if multi_person_disabled:
                    for pose in accepted:
                        draw_pose(frame, pose, None, MIN_KPT_SCORE, disabled=True)
                else:
                    result_pose_ids = {id(pose) for pose, _ in results}
                    for pose, state in results:
                        draw_pose(frame, pose, state, MIN_KPT_SCORE)

                    # Draw low-score or untracked poses faintly if useful.
                    for pose in poses:
                        if id(pose) not in result_pose_ids and pose.score >= 0.02:
                            draw_pose(frame, pose, None, MIN_KPT_SCORE, disabled=True)

                draw_hud(
                    frame,
                    fall_detected,
                    fps_display,
                    accepted_people_count,
                    model_info,
                    disabled_reason=disabled_reason,
                )

                cv2.imshow("MoveNet Fall Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        source.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())