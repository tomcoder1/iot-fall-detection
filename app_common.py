from __future__ import annotations

from time import monotonic
from typing import Protocol

import cv2

from drawing import draw_hud, draw_person
from fall_detector import FallDetector
from pose_types import Person
from settings import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DRAW_ALL_PEOPLE,
    DRAW_HUD,
    DRAW_SKELETON,
    MIN_BODY_AREA_RATIO,
    MIN_PERSON_SCORE,
    MIN_VALID_KEYPOINTS,
    MULTI_PERSON_CONFIRM_FRAMES,
    WINDOW_TITLE,
)


class PoseDetector(Protocol):
    model_name: str

    def detect(self, frame) -> list[Person]:
        ...


def run_app(pose_detector: PoseDetector) -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    fall_detector = FallDetector()

    previous_time = monotonic()
    fps = 0.0
    multi_person_counter = 0

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            break

        now = monotonic()
        dt = max(1e-6, now - previous_time)
        previous_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

        frame_height, frame_width = frame.shape[:2]

        people = pose_detector.detect(frame)

        accepted_people = [
            person
            for person in people
            if _is_accepted_person(person, frame_width, frame_height)
        ]

        disabled_reason = ""

        if len(accepted_people) > 1:
            multi_person_counter += 1
        else:
            multi_person_counter = 0

        multi_person_confirmed = multi_person_counter >= MULTI_PERSON_CONFIRM_FRAMES

        if multi_person_confirmed:
            selected_person = None
            fall_detector.reset()
            result = fall_detector.update(None, frame_width, frame_height)
            disabled_reason = "MULTI-PERSON: DETECTION STOPPED"
        elif len(accepted_people) == 1:
            selected_person = accepted_people[0]
            result = fall_detector.update(selected_person, frame_width, frame_height)
        else:
            selected_person = None
            result = fall_detector.update(None, frame_width, frame_height)

        if DRAW_SKELETON:
            if DRAW_ALL_PEOPLE:
                for person in accepted_people:
                    draw_person(frame, person, selected=person is selected_person)
            elif selected_person is not None:
                draw_person(frame, selected_person, selected=True)

        if DRAW_HUD:
            draw_hud(
                frame=frame,
                result=result,
                fps=fps,
                people_count=len(accepted_people),
                model_name=pose_detector.model_name,
                disabled_reason=disabled_reason,
            )

        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def _is_accepted_person(
    person: Person,
    frame_width: int,
    frame_height: int,
) -> bool:
    if person.score < MIN_PERSON_SCORE:
        return False

    if person.valid_keypoints < MIN_VALID_KEYPOINTS:
        return False

    frame_area = max(1.0, float(frame_width * frame_height))
    body_area = float(person.width * person.height)
    body_area_ratio = body_area / frame_area

    if body_area_ratio < MIN_BODY_AREA_RATIO:
        return False

    return True