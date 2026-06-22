import argparse
import json
from pathlib import Path

from classifier_test_common import run_dataset_test
from detectors.fall_classifier import KeypointFallClassifier
from detectors.pi4_coral_posenet_fall import (
    CLASSIFIER_CONFIG,
    FALL_CLASSIFIER_PATH,
    MODEL_PATH,
    PROJECT_POSENET_DIR,
    CoralPoseNet,
)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="test all videos, including training videos")
    args = parser.parse_args()
    report = json.loads(Path("train/report_pi.json").read_text(encoding="utf-8"))
    video_keys = None if args.all else set(report["test_videos"])
    model = CoralPoseNet(MODEL_PATH, PROJECT_POSENET_DIR)
    detector = KeypointFallClassifier(
        FALL_CLASSIFIER_PATH, CLASSIFIER_CONFIG, expected_platform="pi"
    )
    return run_dataset_test(
        model,
        detector,
        Path("dataset"),
        Path("fall_classifier_results_pi.csv"),
        "PI CORAL CLASSIFIER RESULTS",
        video_keys,
    )

if __name__ == "__main__":
    raise SystemExit(main())