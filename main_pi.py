from app_common import run_app
from detectors.pi4_coral_posenet_fall import CoralPoseNetDetector
from settings import CORAL_POSENET_MODEL_PATH


def main() -> None:
    detector = CoralPoseNetDetector(CORAL_POSENET_MODEL_PATH)
    run_app(detector)


if __name__ == "__main__":
    main()