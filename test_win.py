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
    model = MoveNetMultiPose(resolve_model_path(), NUM_THREADS, INPUT_SIZE)
    detector = KeypointFallClassifier(FALL_CLASSIFIER_PATH, CLASSIFIER_CONFIG)
    return run_dataset_test(
        model,
        detector,
        Path("dataset"),
        Path("fall_classifier_results_win.csv"),
        "WINDOWS CLASSIFIER RESULTS",
    )


if __name__ == "__main__":
    raise SystemExit(main())
