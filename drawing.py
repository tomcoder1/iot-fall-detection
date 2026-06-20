from __future__ import annotations

import cv2
import numpy as np

from fall_detector import FallResult
from pose_types import Person, SKELETON_EDGES
from settings import MIN_KEYPOINT_SCORE


def draw_person(frame: np.ndarray, person: Person, selected: bool = False) -> None:
    color = (0, 255, 0) if selected else (120, 120, 120)

    x1, y1, x2, y2 = person.bbox
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        2,
    )

    for a, b in SKELETON_EDGES:
        ax, ay, ascore = person.keypoints[a]
        bx, by, bscore = person.keypoints[b]

        if ascore >= MIN_KEYPOINT_SCORE and bscore >= MIN_KEYPOINT_SCORE:
            cv2.line(
                frame,
                (int(ax), int(ay)),
                (int(bx), int(by)),
                color,
                2,
            )

    for x, y, score in person.keypoints:
        if score >= MIN_KEYPOINT_SCORE:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)


def draw_hud(
    frame: np.ndarray,
    result: FallResult,
    fps: float,
    people_count: int,
    model_name: str,
    disabled_reason: str = "",
) -> None:
    if disabled_reason:
        status_text = disabled_reason
        status_color = (0, 165, 255)
    elif result.fall_detected:
        status_text = "FALL DETECTED"
        status_color = (0, 0, 255)
    elif result.status == "POSSIBLE_FALL":
        status_text = "POSSIBLE FALL"
        status_color = (0, 165, 255)
    elif result.lying_detected:
        status_text = "LYING"
        status_color = (0, 255, 255)
    elif result.status == "MOVING":
        status_text = "MOVING"
        status_color = (255, 255, 255)
    else:
        status_text = "NORMAL"
        status_color = (0, 255, 0)

    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 128), (0, 0, 0), -1)

    cv2.putText(
        frame,
        status_text,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        status_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | People: {people_count} | Model: {model_name}",
        (12, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"ratio={result.box_ratio:.2f} angle={result.torso_angle:.0f} "
        f"hip_v={result.hip_speed:.2f} shoulder_v={result.shoulder_speed:.2f}",
        (12, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"fall_cnt={result.fall_counter} lying_cnt={result.lying_counter} "
        f"recent_motion={result.recent_motion}",
        (12, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )