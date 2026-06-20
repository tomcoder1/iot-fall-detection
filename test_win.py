from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

import fall_detector as fall_module
from app_common import _is_accepted_person
from detectors.movenet_multipose import MoveNetMultiPoseDetector
from fall_detector import FallDetector
from settings import MOVENET_MODEL_PATH, MULTI_PERSON_CONFIRM_FRAMES


DATASET_DIR = Path("dataset")

# Process every frame. Set to 2 or 3 if testing is too slow.
FRAME_STRIDE = 1

# Set to None to test everything.
# Example: MAX_VIDEOS = 10 for a quick test.
MAX_VIDEOS: Optional[int] = None

# Print each video result.
VERBOSE = True


@dataclass
class VideoResult:
    path: Path
    true_fall: bool
    predicted_fall: bool
    frames_read: int
    frames_used: int
    max_hip_speed: float
    max_shoulder_speed: float
    max_ratio: float
    min_torso_angle: float
    final_status: str


class SimulatedClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def set_time(self, frame_index: int, fps: float) -> None:
        self.t = frame_index / max(fps, 1e-6)


def find_videos(dataset_dir: Path) -> list[Path]:
    videos: list[Path] = []

    for label_folder in ["Fall", "ADL"]:
        videos.extend(dataset_dir.glob(f"**/{label_folder}/*.mp4"))
        videos.extend(dataset_dir.glob(f"**/{label_folder}/*.avi"))
        videos.extend(dataset_dir.glob(f"**/{label_folder}/*.mov"))
        videos.extend(dataset_dir.glob(f"**/{label_folder}/*.mkv"))

    return sorted(videos)


def label_from_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}

    if "fall" in parts:
        return True

    if "adl" in parts:
        return False

    raise ValueError(f"Cannot infer label from path: {path}")


def evaluate_video(
    path: Path,
    pose_detector: MoveNetMultiPoseDetector,
    clock: SimulatedClock,
) -> VideoResult:
    true_fall = label_from_path(path)

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 1 or fps > 240:
        fps = 30.0

    fall_detector = FallDetector()

    frame_index = 0
    frames_used = 0
    multi_person_counter = 0

    predicted_fall = False
    final_status = "NO PERSON"

    max_hip_speed = 0.0
    max_shoulder_speed = 0.0
    max_ratio = 0.0
    min_torso_angle = 999.0

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            break

        frame_index += 1

        if frame_index % FRAME_STRIDE != 0:
            continue

        frames_used += 1
        clock.set_time(frame_index, fps)

        frame_height, frame_width = frame.shape[:2]

        people = pose_detector.detect(frame)

        accepted_people = [
            person
            for person in people
            if _is_accepted_person(person, frame_width, frame_height)
        ]

        if len(accepted_people) > 1:
            multi_person_counter += 1
        else:
            multi_person_counter = 0

        multi_person_confirmed = multi_person_counter >= MULTI_PERSON_CONFIRM_FRAMES

        if multi_person_confirmed:
            fall_detector.reset()
            result = fall_detector.update(None, frame_width, frame_height)
        elif len(accepted_people) == 1:
            result = fall_detector.update(accepted_people[0], frame_width, frame_height)
        else:
            result = fall_detector.update(None, frame_width, frame_height)

        final_status = result.status

        max_hip_speed = max(max_hip_speed, result.hip_speed)
        max_shoulder_speed = max(max_shoulder_speed, result.shoulder_speed)
        max_ratio = max(max_ratio, result.box_ratio)
        min_torso_angle = min(min_torso_angle, result.torso_angle)

        if result.fall_detected:
            predicted_fall = True

    cap.release()

    if min_torso_angle == 999.0:
        min_torso_angle = 0.0

    return VideoResult(
        path=path,
        true_fall=true_fall,
        predicted_fall=predicted_fall,
        frames_read=frame_index,
        frames_used=frames_used,
        max_hip_speed=max_hip_speed,
        max_shoulder_speed=max_shoulder_speed,
        max_ratio=max_ratio,
        min_torso_angle=min_torso_angle,
        final_status=final_status,
    )


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Missing dataset folder: {DATASET_DIR.resolve()}\n"
            "Expected structure like: dataset/Subject 1/Fall/*.mp4 and dataset/Subject 1/ADL/*.mp4"
        )

    videos = find_videos(DATASET_DIR)

    if MAX_VIDEOS is not None:
        videos = videos[:MAX_VIDEOS]

    if not videos:
        raise RuntimeError(
            f"No videos found in {DATASET_DIR.resolve()}.\n"
            "Expected videos inside Fall and ADL folders."
        )

    print(f"Found {len(videos)} videos.")
    print(f"Loading model: {MOVENET_MODEL_PATH}")

    pose_detector = MoveNetMultiPoseDetector(MOVENET_MODEL_PATH)

    # Important:
    # FallDetector uses monotonic() internally.
    # For video testing, we patch it to use video time instead of real wall time.
    clock = SimulatedClock()
    fall_module.monotonic = clock.now

    results: list[VideoResult] = []

    for index, video_path in enumerate(videos, start=1):
        try:
            result = evaluate_video(video_path, pose_detector, clock)
            results.append(result)

            if VERBOSE:
                true_label = "FALL" if result.true_fall else "ADL"
                pred_label = "FALL" if result.predicted_fall else "NO FALL"

                print(
                    f"[{index:03d}/{len(videos):03d}] "
                    f"true={true_label:4s} pred={pred_label:7s} "
                    f"frames={result.frames_used:4d} "
                    f"hip_v={result.max_hip_speed:.2f} "
                    f"shoulder_v={result.max_shoulder_speed:.2f} "
                    f"ratio={result.max_ratio:.2f} "
                    f"angle={result.min_torso_angle:.0f} "
                    f"{result.path}"
                )

        except Exception as exc:
            print(f"[ERROR] {video_path}: {exc}")

    tp = sum(1 for r in results if r.true_fall and r.predicted_fall)
    fn = sum(1 for r in results if r.true_fall and not r.predicted_fall)
    fp = sum(1 for r in results if not r.true_fall and r.predicted_fall)
    tn = sum(1 for r in results if not r.true_fall and not r.predicted_fall)

    total = tp + tn + fp + fn

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    false_positives = [r for r in results if not r.true_fall and r.predicted_fall]
    false_negatives = [r for r in results if r.true_fall and not r.predicted_fall]

    print()
    print("========== RESULTS ==========")
    print(f"Total videos: {total}")
    print()
    print("Confusion matrix:")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    print()
    print(f"Accuracy:    {accuracy:.3f}")
    print(f"Precision:   {precision:.3f}")
    print(f"Recall:      {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 score:    {f1:.3f}")

    print()
    print("False positives:")
    if false_positives:
        for r in false_positives:
            print(f"  FP  {r.path}")
    else:
        print("  none")

    print()
    print("False negatives:")
    if false_negatives:
        for r in false_negatives:
            print(f"  FN  {r.path}")
    else:
        print("  none")


if __name__ == "__main__":
    main()