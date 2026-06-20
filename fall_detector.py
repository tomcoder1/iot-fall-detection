from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from time import monotonic
from typing import Optional, Tuple

import numpy as np

from pose_types import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    Person,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    midpoint,
)
from settings import (
    ALARM_HOLD_SECONDS,
    ALLOW_STATIC_LYING,
    BED_TOP_Y_RATIO,
    FALL_CONFIRM_FRAMES,
    FALL_DROP_SPEED,
    FALL_MOTION_MEMORY_SECONDS,
    HIP_DROP_SPEED,
    HORIZONTAL_TORSO_DEGREES,
    LYING_BOX_RATIO,
    LYING_CONFIRM_FRAMES,
    MIN_KEYPOINT_SCORE,
    PAIR_CLOSE_Y_RATIO,
    PAIR_FAR_X_RATIO,
    SHOULDER_DROP_SPEED,
)


UPPER_BODY = [
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
]

LOWER_BODY = [
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]


@dataclass
class FallResult:
    status: str
    fall_detected: bool
    lying_detected: bool
    reason: str

    box_ratio: float
    torso_angle: float
    recent_drop: float

    hip_speed: float = 0.0
    shoulder_speed: float = 0.0
    fall_counter: int = 0
    lying_counter: int = 0
    recent_motion: bool = False
    lying_candidate: bool = False
    fall_candidate: bool = False


class FallDetector:
    def __init__(self) -> None:
        self.prev_hip_y: Optional[float] = None
        self.prev_shoulder_y: Optional[float] = None
        self.prev_body_center_y: Optional[float] = None
        self.prev_time: Optional[float] = None

        self.recent_motion_until = 0.0
        self.alarm_until = 0.0

        self.fall_counter = 0
        self.lying_counter = 0
        self.was_lying = False

    def reset(self) -> None:
        self.prev_hip_y = None
        self.prev_shoulder_y = None
        self.prev_body_center_y = None
        self.prev_time = None

        self.recent_motion_until = 0.0
        self.alarm_until = 0.0

        self.fall_counter = 0
        self.lying_counter = 0
        self.was_lying = False

    def update(
        self,
        person: Optional[Person],
        frame_width: int,
        frame_height: int,
    ) -> FallResult:
        now = monotonic()

        if person is None:
            self.reset()
            return FallResult(
                status="NO PERSON",
                fall_detected=False,
                lying_detected=False,
                reason="no accepted person",
                box_ratio=0.0,
                torso_angle=90.0,
                recent_drop=0.0,
            )

        dt = None
        if self.prev_time is not None:
            dt = max(1e-3, now - self.prev_time)

        box_ratio = person.width / person.height
        body_height = max(1.0, person.height)

        torso_angle = self._torso_angle_degrees(person)

        hip = midpoint(person, LEFT_HIP, RIGHT_HIP, MIN_KEYPOINT_SCORE)
        shoulder = midpoint(person, LEFT_SHOULDER, RIGHT_SHOULDER, MIN_KEYPOINT_SCORE)

        hip_speed = 0.0
        shoulder_speed = 0.0

        if dt is not None and hip is not None and self.prev_hip_y is not None:
            hip_speed = ((hip[1] - self.prev_hip_y) / dt) / body_height

        if dt is not None and shoulder is not None and self.prev_shoulder_y is not None:
            shoulder_speed = ((shoulder[1] - self.prev_shoulder_y) / dt) / body_height

        body_center_y = self._body_center_y(person)
        recent_drop = 0.0

        if body_center_y is not None and self.prev_body_center_y is not None:
            recent_drop = max(0.0, body_center_y - self.prev_body_center_y)

        angle_rule = torso_angle <= HORIZONTAL_TORSO_DEGREES
        ratio_rule = box_ratio >= LYING_BOX_RATIO
        pair_rule = self._upper_lower_pair_rule(person, frame_width, frame_height)

        geometry_lying = angle_rule and ratio_rule
        lying_candidate = geometry_lying or (ratio_rule and pair_rule)

        bed_rule = self._is_on_bed_region(person, frame_height)

        prev_was_lying = self.was_lying

        sudden_motion = (
            hip_speed >= HIP_DROP_SPEED
            or shoulder_speed >= SHOULDER_DROP_SPEED
        )

        strong_fall_motion = (
            hip_speed >= FALL_DROP_SPEED
            or shoulder_speed >= FALL_DROP_SPEED
        )

        if strong_fall_motion and not prev_was_lying:
            self.recent_motion_until = now + FALL_MOTION_MEMORY_SECONDS

        recent_motion = now <= self.recent_motion_until

        fall_candidate = (
            lying_candidate
            and recent_motion
            and (not prev_was_lying or self.fall_counter > 0)
        )

        if ALLOW_STATIC_LYING and lying_candidate:
            fall_candidate = True

        if bed_rule:
            lying_candidate = False
            fall_candidate = False

        if lying_candidate:
            self.lying_counter += 1
        else:
            self.lying_counter = max(0, self.lying_counter - 1)

        if fall_candidate:
            self.fall_counter = min(FALL_CONFIRM_FRAMES, self.fall_counter + 1)
        else:
            self.fall_counter = max(0, self.fall_counter - 1)

        lying_confirmed = self.lying_counter >= LYING_CONFIRM_FRAMES
        confirmed_fall = self.fall_counter >= FALL_CONFIRM_FRAMES

        if confirmed_fall:
            self.alarm_until = now + ALARM_HOLD_SECONDS

        fall_detected = now <= self.alarm_until

        if fall_detected:
            status = "FALL"
            reason = "lying posture after strong downward hip/shoulder motion"
        elif self.fall_counter > 0:
            status = "POSSIBLE_FALL"
            reason = "fall-like motion seen, waiting for confirmation"
        elif lying_confirmed:
            status = "LYING"
            reason = "lying posture without enough fall motion"
        elif sudden_motion:
            status = "MOVING"
            reason = "downward motion seen, but posture is not lying"
        else:
            status = "NORMAL"
            reason = "standing or moving normally"

        self.was_lying = bool(lying_candidate)

        if hip is not None:
            self.prev_hip_y = hip[1]

        if shoulder is not None:
            self.prev_shoulder_y = shoulder[1]

        if body_center_y is not None:
            self.prev_body_center_y = body_center_y

        self.prev_time = now

        return FallResult(
            status=status,
            fall_detected=fall_detected,
            lying_detected=lying_confirmed,
            reason=reason,
            box_ratio=box_ratio,
            torso_angle=torso_angle,
            recent_drop=recent_drop,
            hip_speed=hip_speed,
            shoulder_speed=shoulder_speed,
            fall_counter=self.fall_counter,
            lying_counter=self.lying_counter,
            recent_motion=recent_motion,
            lying_candidate=lying_candidate,
            fall_candidate=fall_candidate,
        )

    def _body_center_y(self, person: Person) -> Optional[float]:
        shoulder_mid = midpoint(person, LEFT_SHOULDER, RIGHT_SHOULDER, MIN_KEYPOINT_SCORE)
        hip_mid = midpoint(person, LEFT_HIP, RIGHT_HIP, MIN_KEYPOINT_SCORE)

        points = [p for p in [shoulder_mid, hip_mid] if p is not None]

        if not points:
            return None

        return float(sum(p[1] for p in points) / len(points))

    def _torso_angle_degrees(self, person: Person) -> float:
        shoulder_mid = midpoint(person, LEFT_SHOULDER, RIGHT_SHOULDER, MIN_KEYPOINT_SCORE)
        hip_mid = midpoint(person, LEFT_HIP, RIGHT_HIP, MIN_KEYPOINT_SCORE)

        if shoulder_mid is None or hip_mid is None:
            return 90.0

        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]

        return abs(degrees(atan2(abs(dy), abs(dx) + 1e-6)))

    def _upper_lower_pair_rule(
        self,
        person: Person,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        upper_points = self._valid_points(person, UPPER_BODY)
        lower_points = self._valid_points(person, LOWER_BODY)

        if len(upper_points) == 0 or len(lower_points) == 0:
            return False

        for upper_x, upper_y in upper_points:
            for lower_x, lower_y in lower_points:
                y_close = abs(upper_y - lower_y) <= frame_height * PAIR_CLOSE_Y_RATIO
                x_far = abs(upper_x - lower_x) >= frame_width * PAIR_FAR_X_RATIO

                if y_close and x_far:
                    return True

        return False

    def _is_on_bed_region(self, person: Person, frame_height: int) -> bool:
        if BED_TOP_Y_RATIO is None:
            return False

        upper_points = self._valid_points(person, UPPER_BODY)

        if len(upper_points) == 0:
            return False

        upper_y_values = [p[1] for p in upper_points]
        median_upper_y = float(np.median(upper_y_values))

        return median_upper_y > frame_height * float(BED_TOP_Y_RATIO)

    def _valid_points(
        self,
        person: Person,
        indices: list[int],
    ) -> list[Tuple[float, float]]:
        points: list[Tuple[float, float]] = []

        for index in indices:
            x, y, score = person.keypoints[index]

            if score >= MIN_KEYPOINT_SCORE:
                points.append((float(x), float(y)))

        return points