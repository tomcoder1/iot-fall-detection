from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

# ============================================================
# User settings. Edit these, do not use command-line arguments.
# ============================================================

MODEL_PATH = Path("movenet_multipose_lightning.tflite")
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
DISPLAY = True
MIRROR_IMAGE = False
DEBUG_EVERY_N_FRAMES = 30
NUM_THREADS = 4
INPUT_SIZE = 256

# Pose filtering. MoveNet MultiPose gives up to 6 people.
MIN_POSE_SCORE = 0.10
MIN_KPT_SCORE = 0.08
MIN_VALID_KEYPOINTS = 5
MIN_BODY_AREA = 0.015

# If more than one accepted person appears, fall detection stops.
STOP_WHEN_MULTIPLE_PEOPLE = True
MULTI_PERSON_CONFIRM_FRAMES = 2

# Fall geometry thresholds.
ANGLE_THRESHOLD = 60.0
RATIO_THRESHOLD = 0.85
PAIR_THRESHOLD_Y = 0.12
PAIR_THRESHOLD_X = 0.25

# Motion thresholds. Normalized by body height per second.
HIP_DROP_SPEED = 0.35
SHOULDER_DROP_SPEED = 0.35
FALL_DROP_SPEED = 1.00
FALL_MOTION_MEMORY_SEC = 0.90
FALL_FRAMES = 2
ALARM_HOLD_SEC = 5.0

# Keep False for real detection. Slow lying should become LYING, not FALL.
ALLOW_STATIC_LYING = False

# Optional bed/sofa cancellation line. Example: 0.55 means upper body below
# 55 percent of image height will not trigger fall. Keep None unless calibrated.
BED_TOP_Y: Optional[float] = None

# Public boolean for IoT logic.
fall_detected = False
fall_alarm_until = 0.0

# ============================================================
# Skeleton constants. Same 17-keypoint order as PoseNet / MoveNet.
# ============================================================

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
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

UPPER_BODY = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
LOWER_BODY = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]

SKELETON_EDGES = [
    (NOSE, LEFT_SHOULDER), (NOSE, RIGHT_SHOULDER),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]


@dataclass
class Pose:
    keypoints: np.ndarray  # [17, 3], normalized y, x, score
    bbox: Tuple[float, float, float, float]
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


# ============================================================
# TFLite MoveNet adapter
# ============================================================

def safe_int_list(values: Iterable[object]) -> List[int]:
    return [int(v) for v in values]


def letterbox_bgr_to_rgb(frame_bgr: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, Dict[str, float]]:
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


class MoveNetMultiPose:
    def __init__(self, model_path: Path, num_threads: int = 4, force_input_size: int = 256) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
            runtime_name = "tflite_runtime"
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
                runtime_name = "tensorflow.lite"
            except ImportError as exc:
                raise RuntimeError(
                    "Could not import TFLite interpreter. On Windows, install TensorFlow:\n"
                    "py -m pip install tensorflow"
                ) from exc

        self.interpreter = Interpreter(model_path=str(model_path), num_threads=num_threads)
        input_details = self.interpreter.get_input_details()
        pre_shape = safe_int_list(input_details[0]["shape"])
        input_index = int(input_details[0]["index"])

        if len(pre_shape) == 4 and (pre_shape[1] <= 1 or pre_shape[2] <= 1):
            self.interpreter.resize_tensor_input(input_index, [1, force_input_size, force_input_size, 3], strict=False)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = int(self.input_details[0]["index"])
        self.input_shape = safe_int_list(self.input_details[0]["shape"])
        self.input_dtype = self.input_details[0]["dtype"]
        self.input_quantization = self.input_details[0].get("quantization", (0.0, 0))
        self.input_height = int(self.input_shape[1])
        self.input_width = int(self.input_shape[2])

        print("[MODEL] runtime:", runtime_name)
        print("[MODEL] path:", model_path)
        print("[MODEL] input shape:", self.input_shape)
        print("[MODEL] input dtype:", self.input_dtype)

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
        return tensor.astype(self.input_dtype)

    def infer(self, frame_bgr: np.ndarray) -> Tuple[List[Pose], Dict[str, object]]:
        input_rgb, meta = letterbox_bgr_to_rgb(frame_bgr, self.input_width, self.input_height)
        tensor = self._prepare_tensor(input_rgb)
        start = time.time()
        self.interpreter.set_tensor(self.input_index, tensor)
        self.interpreter.invoke()
        inference_ms = (time.time() - start) * 1000.0
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        poses = parse_movenet_multipose(output, meta)
        return poses, {
            "model_type": "movenet_multipose_cpu",
            "inference_ms": float(inference_ms),
            "raw_pose_scores": [float(p.score) for p in poses],
        }


def parse_movenet_multipose(output: np.ndarray, meta: Dict[str, float]) -> List[Pose]:
    arr = np.squeeze(np.asarray(output))
    if arr.ndim == 1 and arr.size == 56:
        arr = arr.reshape(1, 56)
    if arr.ndim != 2 or arr.shape[-1] != 56:
        raise RuntimeError(f"Unexpected MoveNet MultiPose output shape: raw={output.shape}, squeezed={arr.shape}")

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
        bbox = (min(ymin, ymax), min(xmin, xmax), max(ymin, ymax), max(xmin, xmax))
        poses.append(Pose(keypoints=converted, bbox=bbox, score=pose_score))
    return poses


# ============================================================
# Geometry helpers
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


def midpoint(keypoints: np.ndarray, left_idx: int, right_idx: int, min_score: float) -> Optional[Tuple[float, float]]:
    points = []
    for idx in (left_idx, right_idx):
        y, x, score = keypoints[idx]
        if score >= min_score:
            points.append((float(y), float(x)))
    if not points:
        return None
    return float(np.mean([p[0] for p in points])), float(np.mean([p[1] for p in points]))


def pose_bbox_from_keypoints(keypoints: np.ndarray, min_score: float) -> Optional[Tuple[float, float, float, float]]:
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


def center_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    ymin, xmin, ymax, xmax = bbox
    return float((ymin + ymax) / 2.0), float((xmin + xmax) / 2.0)


def torso_angle_degrees(keypoints: np.ndarray, min_score: float) -> Optional[float]:
    shoulder = midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, min_score)
    hip = midpoint(keypoints, LEFT_HIP, RIGHT_HIP, min_score)
    if shoulder is None or hip is None:
        return None
    dy = abs(shoulder[0] - hip[0])
    dx = abs(shoulder[1] - hip[1])
    return float(math.degrees(math.atan2(dy, dx + 1e-6)))


def pair_rule_upper_lower(keypoints: np.ndarray, min_score: float, threshold_y: float, threshold_x: float) -> bool:
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


def is_on_bed_region(keypoints: np.ndarray, min_score: float, bed_top_y: Optional[float]) -> bool:
    if bed_top_y is None:
        return False
    upper = valid_points(keypoints, UPPER_BODY, min_score)
    if len(upper) == 0:
        return False
    return float(np.median(upper[:, 0])) > bed_top_y


def enough_pose_quality(pose: Pose, min_kpt_score: float) -> bool:
    valid_count = int(np.sum(pose.keypoints[:, 2] >= min_kpt_score))
    if valid_count < MIN_VALID_KEYPOINTS:
        return False
    bbox = pose_bbox_from_keypoints(pose.keypoints, min_kpt_score) or pose.bbox
    _, width, height = bbox_ratio(bbox)
    if width * height < MIN_BODY_AREA:
        return False
    return True


# ============================================================
# Fall detector
# ============================================================

class FallDetector:
    def __init__(self) -> None:
        self.states: Dict[int, PersonState] = {}
        self.next_track_id = 1
        self.track_distance = 0.30

    def reset(self) -> None:
        self.states.clear()
        self.next_track_id = 1

    def accepted_poses(self, poses: List[Pose]) -> List[Pose]:
        candidates = [p for p in poses if p.score >= MIN_POSE_SCORE and enough_pose_quality(p, MIN_KPT_SCORE)]
        candidates.sort(key=lambda p: p.score, reverse=True)
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

        for track_id in [tid for tid, st in self.states.items() if now - st.last_seen > 3.0]:
            del self.states[track_id]

        return results

    def _assign_track(self, pose: Pose, now: float, assigned_tracks: set, candidate_count: int) -> PersonState:
        bbox = pose_bbox_from_keypoints(pose.keypoints, MIN_KPT_SCORE) or pose.bbox
        center = center_from_bbox(bbox)

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
        bbox = pose_bbox_from_keypoints(keypoints, MIN_KPT_SCORE) or pose.bbox
        ratio, body_w, body_h = bbox_ratio(bbox)

        angle = torso_angle_degrees(keypoints, MIN_KPT_SCORE)
        angle_rule = angle is not None and angle < ANGLE_THRESHOLD
        ratio_rule = ratio > RATIO_THRESHOLD
        pair_rule = pair_rule_upper_lower(keypoints, MIN_KPT_SCORE, PAIR_THRESHOLD_Y, PAIR_THRESHOLD_X)
        bed_rule = is_on_bed_region(keypoints, MIN_KPT_SCORE, BED_TOP_Y)

        hip = midpoint(keypoints, LEFT_HIP, RIGHT_HIP, MIN_KPT_SCORE)
        shoulder = midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, MIN_KPT_SCORE)

        hip_speed = 0.0
        shoulder_speed = 0.0
        dt = max(1e-3, now - state.prev_time) if state.prev_time is not None else None

        if hip is not None and state.prev_hip_y is not None and dt is not None:
            hip_speed = ((hip[0] - state.prev_hip_y) / dt) / max(body_h, 1e-6)
        if shoulder is not None and state.prev_shoulder_y is not None and dt is not None:
            shoulder_speed = ((shoulder[0] - state.prev_shoulder_y) / dt) / max(body_h, 1e-6)

        prev_was_lying = state.was_lying
        sudden_motion = hip_speed > HIP_DROP_SPEED or shoulder_speed > SHOULDER_DROP_SPEED
        strong_fall_motion = hip_speed > FALL_DROP_SPEED or shoulder_speed > FALL_DROP_SPEED

        if strong_fall_motion and not prev_was_lying:
            state.recent_motion_until = now + FALL_MOTION_MEMORY_SEC
        recent_motion = now <= state.recent_motion_until

        geometry_lying = angle_rule and ratio_rule
        lying_candidate = geometry_lying or (ratio_rule and pair_rule)

        fall_candidate = lying_candidate and recent_motion and (not prev_was_lying or state.fall_counter > 0)
        if ALLOW_STATIC_LYING and lying_candidate:
            fall_candidate = True

        if bed_rule:
            fall_candidate = False
            lying_candidate = False

        if fall_candidate:
            state.fall_counter = min(FALL_FRAMES, state.fall_counter + 1)
        else:
            state.fall_counter = max(0, state.fall_counter - 1)

        if state.fall_counter >= FALL_FRAMES:
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
            "sudden_motion": bool(sudden_motion),
            "strong_fall_motion": bool(strong_fall_motion),
            "recent_motion": bool(recent_motion),
            "lying_candidate": bool(lying_candidate),
            "fall_candidate": bool(fall_candidate),
            "fall_counter": int(state.fall_counter),
        }


# ============================================================
# Drawing
# ============================================================

def draw_pose(frame: np.ndarray, pose: Pose, state: Optional[PersonState], disabled: bool = False) -> None:
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
        ya, xa, sa = keypoints[a]
        yb, xb, sb = keypoints[b]
        if sa >= MIN_KPT_SCORE and sb >= MIN_KPT_SCORE:
            cv2.line(frame, (int(xa * w), int(ya * h)), (int(xb * w), int(yb * h)), color, 2)

    for y, x, score in keypoints:
        if score >= MIN_KPT_SCORE:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 255), -1)

    bbox = pose_bbox_from_keypoints(keypoints, MIN_KPT_SCORE) or pose.bbox
    ymin, xmin, ymax, xmax = bbox
    cv2.rectangle(frame, (int(xmin * w), int(ymin * h)), (int(xmax * w), int(ymax * h)), color, 2)

    if disabled:
        label = "MULTI-PERSON: FALL OFF"
    elif state is None:
        label = f"score={pose.score:.2f}"
    else:
        dbg = state.debug
        max_speed = max(float(dbg.get("hip_speed", 0.0)), float(dbg.get("shoulder_speed", 0.0)))
        label = f"id={state.track_id} {state.last_status} cnt={dbg.get('fall_counter', 0)} r={dbg.get('ratio', 0):.2f} a={dbg.get('angle', -1):.0f} v={max_speed:.2f}"

    cv2.putText(frame, label, (int(xmin * w), max(20, int(ymin * h) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_hud(frame: np.ndarray, current_fall_detected: bool, fps: float, people_count: int, model_info: Dict[str, object], disabled_reason: Optional[str]) -> None:
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


# ============================================================
# Main
# ============================================================

def main() -> int:
    global fall_detected, fall_alarm_until

    model = MoveNetMultiPose(MODEL_PATH, NUM_THREADS, INPUT_SIZE)
    detector = FallDetector()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    if not cap.isOpened():
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

    print("[INFO] Started Windows MoveNet MultiPose fall detector.")
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

            multi_person_disabled = STOP_WHEN_MULTIPLE_PEOPLE and multi_person_hits >= MULTI_PERSON_CONFIRM_FRAMES
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
                    fall_alarm_until = now + ALARM_HOLD_SEC
                fall_detected = now <= fall_alarm_until

            processed_frames += 1
            elapsed = max(1e-6, time.time() - start_time)
            fps = processed_frames / elapsed

            if DEBUG_EVERY_N_FRAMES and frame_idx % DEBUG_EVERY_N_FRAMES == 0:
                print(f"[DEBUG] frame={frame_idx} fps={fps:.1f} people={people_count} fall_detected={fall_detected} scores={[round(float(p.score), 2) for p in accepted]}")

            if DISPLAY:
                if multi_person_disabled:
                    for pose in accepted:
                        draw_pose(frame, pose, None, disabled=True)
                else:
                    result_pose_ids = {id(pose) for pose, _ in results}
                    for pose, state in results:
                        draw_pose(frame, pose, state)
                    for pose in accepted:
                        if id(pose) not in result_pose_ids:
                            draw_pose(frame, pose, None, disabled=True)

                draw_hud(frame, fall_detected, fps, people_count, model_info, disabled_reason)
                cv2.imshow("Windows MoveNet MultiPose Fall Detection", frame)
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
