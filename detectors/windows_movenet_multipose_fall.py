from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from app_common import AppOptions, run_app

from .fall_core import (
    FallConfig,
    Pose,
)

# ============================================================
# User settings. Edit these, do not use command-line arguments.
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/movenet_multipose_lightning.tflite"
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

    fall_frames=4,
    high_confidence_increment=2,
    alarm_hold_sec=5.0,

    allow_static_lying=False,
    allow_no_upright_if_very_fast=True,
    very_fast_drop_speed=2.50,

    bed_top_y=None,
)

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


MoveNetMultiPoseDetector = MoveNetMultiPose


def main() -> int:
    model = MoveNetMultiPose(resolve_model_path(), NUM_THREADS, INPUT_SIZE)
    options = AppOptions(
        title="Windows MoveNet MultiPose Fall Detection",
        camera_index=CAMERA_INDEX,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        camera_fps=CAMERA_FPS,
        display=DISPLAY,
        mirror_image=MIRROR_IMAGE,
        debug_every_n_frames=DEBUG_EVERY_N_FRAMES,
        camera_backend=cv2.CAP_DSHOW,
    )
    return run_app(model, CONFIG, options)


if __name__ == "__main__":
    raise SystemExit(main())
