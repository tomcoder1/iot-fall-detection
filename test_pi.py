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
    model = CoralPoseNet(MODEL_PATH, PROJECT_POSENET_DIR)
    detector = KeypointFallClassifier(FALL_CLASSIFIER_PATH, CLASSIFIER_CONFIG)
    return run_dataset_test(
        model,
        detector,
        Path("dataset"),
        Path("fall_classifier_results_pi.csv"),
        "PI CORAL CLASSIFIER RESULTS",
    )


if __name__ == "__main__":
    raise SystemExit(main())
