from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import psutil


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from detectors.pi4_coral_posenet_fall import (  # noqa: E402
    CLASSIFIER_CONFIG,
    FALL_CLASSIFIER_PATH,
    MODEL_PATH,
    PROJECT_POSENET_DIR,
    CoralPoseNet,
)
from detectors.fall_classifier import KeypointFallClassifier  # noqa: E402

def cpu_temperature_c() -> float | None:
    try:
        output = subprocess.check_output(
            ["vcgencmd", "measure_temp"], text=True, timeout=2
        ).strip()
        return float(output.split("=")[1].split("'")[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record fall-detector FPS, Pi temperature, and memory once per second."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Video replay used as the frame source")
    source.add_argument("--camera-index", type=int, help="OpenCV camera index")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to record")
    parser.add_argument("--output", type=Path, default=Path("measure/results/runtime.csv"))
    return parser.parse_args()

def main() -> int:
    args = arguments()
    capture = cv2.VideoCapture(
        str(args.video) if args.video is not None else args.camera_index
    )
    if not capture.isOpened():
        raise RuntimeError("Could not open the requested frame source")

    source_name = (
        str(args.video.resolve())
        if args.video is not None
        else f"camera:{args.camera_index}"
    )
    model = CoralPoseNet(MODEL_PATH, PROJECT_POSENET_DIR)
    detector = KeypointFallClassifier(
        FALL_CLASSIFIER_PATH, CLASSIFIER_CONFIG, expected_platform="pi"
    )
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    sample_started = started
    sample_frames = 0

    try:
        while time.perf_counter() - started < args.duration:
            ok, frame = capture.read()
            if not ok or frame is None:
                if args.video is None:
                    raise RuntimeError("Camera stopped returning frames")
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            poses, _ = model.infer(frame)
            accepted = detector.accepted_poses(poses)
            detector.update(accepted[0] if accepted else None, time.perf_counter())
            sample_frames += 1
            now = time.perf_counter()
            sample_elapsed = now - sample_started
            if sample_elapsed >= 1.0:
                rows.append(
                    {
                        "elapsed_s": round(now - started, 3),
                        "fps": round(sample_frames / sample_elapsed, 3),
                        "cpu_temperature_c": cpu_temperature_c(),
                        "memory_percent": round(psutil.virtual_memory().percent, 3),
                    }
                )
                sample_started = now
                sample_frames = 0
    finally:
        capture.release()

    if not rows:
        raise RuntimeError("No complete one-second samples were collected")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": source_name,
        "duration_s": round(rows[-1]["elapsed_s"], 3),
        "samples": len(rows),
        "average_fps": sum(row["fps"] for row in rows) / len(rows),
        "peak_temperature_c": max(
            row["cpu_temperature_c"]
            for row in rows
            if row["cpu_temperature_c"] is not None
        ),
        "average_memory_percent": sum(row["memory_percent"] for row in rows)
        / len(rows),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())