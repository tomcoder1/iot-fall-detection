from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .fall_core import (
    FallConfig,
    FallDetector,
    LEFT_HIP,
    LEFT_SHOULDER,
    Pose,
    PersonState,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    SKELETON_EDGES,
    bbox_ratio,
    center_from_bbox,
    pose_bbox_from_keypoints,
)

# ============================================================
# User settings. Edit these, do not use command-line arguments.
# ============================================================
MODEL_PATH = Path("models/movenet_multipose_lightning.tflite")
MODEL_PATH_FALLBACK = Path("movenet_multipose_lightning.tflite")
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
DISPLAY = True
MIRROR_IMAGE = False
DEBUG_EVERY_N_FRAMES = 30
NUM_THREADS = 4
INPUT_SIZE = 256

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


def safe_int_list(values: Iterable[object]) -> List[int]:
    return [int(v) for v in values]


def resolve_model_path() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    if MODEL_PATH_FALLBACK.exists():
        return MODEL_PATH_FALLBACK
    raise FileNotFoundError(f"Missing model. Tried {MODEL_PATH} and {MODEL_PATH_FALLBACK}")


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
            self.interpreter.resize_tensor_input(
                input_index,
                [1, force_input_size, force_input_size, 3],
                strict=False,
            )

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
    global fall_detected, fall_alarm_until

    model = MoveNetMultiPose(resolve_model_path(), NUM_THREADS, INPUT_SIZE)
    detector = FallDetector(CONFIG)

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

            if DEBUG_EVERY_N_FRAMES and frame_idx % DEBUG_EVERY_N_FRAMES == 0:
                debug_states = [st.debug for _, st in results]
                print(
                    f"[DEBUG] frame={frame_idx} fps={fps:.1f} people={people_count} "
                    f"fall_detected={fall_detected} states={debug_states}"
                )

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
