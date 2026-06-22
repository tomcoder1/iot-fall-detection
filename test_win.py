import argparse
import json
from pathlib import Path

from classifier_test_common import run_dataset_test
from detectors.fall_classifier import KeypointFallClassifier
from detectors.windows_movenet_multipose_fall import (
    CLASSIFIER_CONFIG,
    FALL_CLASSIFIER_PATH,
    INPUT_SIZE,
    NUM_THREADS,
    MoveNetMultiPose,
    resolve_model_path,
)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="test all videos, including training videos")
    args = parser.parse_args()
    report = json.loads(Path("train/report_windows.json").read_text(encoding="utf-8"))
    video_keys = None if args.all else set(report["test_videos"])
    model = MoveNetMultiPose(resolve_model_path(), NUM_THREADS, INPUT_SIZE)
    detector = KeypointFallClassifier(
        FALL_CLASSIFIER_PATH, CLASSIFIER_CONFIG, expected_platform="windows"
    )
    return run_dataset_test(
        model,
        detector,
        Path("dataset"),
        Path("fall_classifier_results_win.csv"),
        "WINDOWS CLASSIFIER RESULTS",
        video_keys,
    )

if __name__ == "__main__":
    raise SystemExit(main())