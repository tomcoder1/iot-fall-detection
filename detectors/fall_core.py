from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

UPPER_BODY = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
LOWER_BODY = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
CORE_BODY = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

SKELETON_EDGES = [
    (NOSE, LEFT_SHOULDER),
    (NOSE, RIGHT_SHOULDER),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]


@dataclass
class Pose:
    keypoints: np.ndarray
    bbox: Tuple[float, float, float, float]
    score: float


@dataclass
class FallConfig:
    # Pose filtering.
    min_pose_score: float = 0.10
    min_kpt_score: float = 0.08
    min_valid_keypoints: int = 5
    min_body_area: float = 0.015

    # Multi-person rule.
    stop_when_multiple_people: bool = True
    multi_person_confirm_frames: int = 2

    # Upright and ground geometry.
    upright_angle: float = 65.0
    upright_max_ratio: float = 1.00

    # Normal horizontal fall rule.
    horizontal_angle: float = 35.0
    horizontal_ratio: float = 1.30

    # Compact fall rule.
    # This catches real falls where the body is flat but the bbox ratio is only around 1.0.
    low_horizontal_angle: float = 35.0
    low_horizontal_ratio: float = 0.95

    pair_horizontal_ratio: float = 1.55
    pair_threshold_y: float = 0.12
    pair_threshold_x: float = 0.25

    # Motion. Speeds are normalized by body height per second.
    fall_drop_speed: float = 0.95
    soft_drop_speed: float = 0.35
    motion_memory_sec: float = 1.75
    descent_timeout_sec: float = 2.25
    upright_memory_sec: float = 5.00

    # Low body check.
    min_low_drop_norm: float = 0.08
    min_low_drop_body_heights: float = 0.25

    # Confirmation.
    fall_frames: int = 4
    high_confidence_increment: int = 2
    alarm_hold_sec: float = 5.0

    allow_static_lying: bool = False
    allow_no_upright_if_very_fast: bool = True
    very_fast_drop_speed: float = 2.50

    # Optional calibrated cancellation line.
    # Example: 0.55 means upper body below 55 percent of image height is treated as bed/sofa region.
    bed_top_y: Optional[float] = None


@dataclass
class PersonState:
    track_id: int
    center: Tuple[float, float]
    last_seen: float

    prev_hip_y: Optional[float] = None
    prev_shoulder_y: Optional[float] = None
    prev_center_y: Optional[float] = None
    prev_time: Optional[float] = None

    last_upright_time: Optional[float] = None
    last_upright_center_y: Optional[float] = None
    recent_motion_until: float = 0.0
    descent_started_at: Optional[float] = None

    fall_counter: int = 0
    ground_counter: int = 0

    was_horizontal: bool = False
    last_status: str = "UNKNOWN"
    debug: Dict[str, Union[float, bool, int, str]] = field(default_factory=dict)


def valid_points(keypoints: np.ndarray, indices: Iterable[int], min_score: float) -> np.ndarray:
    points = []
    for idx in indices:
        y, x, score = keypoints[idx]
        if score >= min_score:
            points.append([float(y), float(x), float(score)])
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def midpoint(
    keypoints: np.ndarray,
    left_idx: int,
    right_idx: int,
    min_score: float,
) -> Optional[Tuple[float, float]]:
    points = []
    for idx in (left_idx, right_idx):
        y, x, score = keypoints[idx]
        if score >= min_score:
            points.append((float(y), float(x)))
    if not points:
        return None
    return float(np.mean([p[0] for p in points])), float(np.mean([p[1] for p in points]))


def pose_bbox_from_keypoints(
    keypoints: np.ndarray,
    min_score: float,
) -> Optional[Tuple[float, float, float, float]]:
    points = valid_points(keypoints, range(17), min_score)
    if len(points) < 4:
        return None
    ymin = float(np.min(points[:, 0]))
    ymax = float(np.max(points[:, 0]))
    xmin = float(np.min(points[:, 1]))
    xmax = float(np.max(points[:, 1]))
    return ymin, xmin, ymax, xmax


def bbox_ratio(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    ymin, xmin, ymax, xmax = bbox
    height = max(1e-6, ymax - ymin)
    width = max(1e-6, xmax - xmin)
    return width / height, width, height


def center_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    ymin, xmin, ymax, xmax = bbox
    return float((ymin + ymax) / 2.0), float((xmin + xmax) / 2.0)


def torso_angle_degrees(keypoints: np.ndarray, min_score: float) -> Optional[float]:
    shoulder = midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, min_score)
    hip = midpoint(keypoints, LEFT_HIP, RIGHT_HIP, min_score)
    if shoulder is None or hip is None:
        return None
    dy = abs(shoulder[0] - hip[0])
    dx = abs(shoulder[1] - hip[1])
    return float(math.degrees(math.atan2(dy, dx + 1e-6)))


def pair_rule_upper_lower(
    keypoints: np.ndarray,
    min_score: float,
    threshold_y: float,
    threshold_x: float,
) -> bool:
    upper = valid_points(keypoints, UPPER_BODY, min_score)
    lower = valid_points(keypoints, LOWER_BODY, min_score)
    if len(upper) == 0 or len(lower) == 0:
        return False

    for upper_y, upper_x, _ in upper:
        for lower_y, lower_x, _ in lower:
            y_close = abs(float(upper_y) - float(lower_y)) <= threshold_y
            x_far = abs(float(upper_x) - float(lower_x)) >= threshold_x
            if y_close and x_far:
                return True
    return False


def is_on_bed_region(keypoints: np.ndarray, min_score: float, bed_top_y: Optional[float]) -> bool:
    if bed_top_y is None:
        return False
    upper = valid_points(keypoints, UPPER_BODY, min_score)
    if len(upper) == 0:
        return False
    return float(np.median(upper[:, 0])) > bed_top_y


def enough_pose_quality(pose: Pose, config: FallConfig) -> bool:
    valid_count = int(np.sum(pose.keypoints[:, 2] >= config.min_kpt_score))
    if valid_count < config.min_valid_keypoints:
        return False
    bbox = pose_bbox_from_keypoints(pose.keypoints, config.min_kpt_score) or pose.bbox
    _, width, height = bbox_ratio(bbox)
    return width * height >= config.min_body_area


class FallDetector:
    """
    Event-based fall detector.

    The previous logic treated a horizontal body shape plus recent motion as a fall.
    This detector requires a sequence: upright context, downward motion, then a
    horizontal/low body posture that persists for several frames.
    """

    def __init__(self, config: Optional[FallConfig] = None) -> None:
        self.config = config or FallConfig()
        self.states: Dict[int, PersonState] = {}
        self.next_track_id = 1
        self.track_distance = 0.30

    def reset(self) -> None:
        self.states.clear()
        self.next_track_id = 1

    def accepted_poses(self, poses: List[Pose]) -> List[Pose]:
        candidates = [
            p for p in poses
            if p.score >= self.config.min_pose_score and enough_pose_quality(p, self.config)
        ]
        candidates.sort(key=lambda p: p.score, reverse=True)
        return candidates

    def update(self, poses: List[Pose], now: float) -> List[Tuple[Pose, PersonState]]:
        results: List[Tuple[Pose, PersonState]] = []
        candidates = self.accepted_poses(poses)
        assigned_tracks = set()

        for pose in candidates:
            state = self._assign_track(pose, now, assigned_tracks, len(candidates))
            assigned_tracks.add(state.track_id)
            self._update_state_with_pose(state, pose, now)
            results.append((pose, state))

        for track_id in [tid for tid, st in self.states.items() if now - st.last_seen > 3.0]:
            del self.states[track_id]

        return results

    def _assign_track(
        self,
        pose: Pose,
        now: float,
        assigned_tracks: set,
        candidate_count: int,
    ) -> PersonState:
        bbox = pose_bbox_from_keypoints(pose.keypoints, self.config.min_kpt_score) or pose.bbox
        center = center_from_bbox(bbox)

        if candidate_count == 1 and len(self.states) == 1 and len(assigned_tracks) == 0:
            state = next(iter(self.states.values()))
            state.center = center
            state.last_seen = now
            return state

        best_id = None
        best_dist = 1e9
        for track_id, state in self.states.items():
            if track_id in assigned_tracks:
                continue
            dist = math.hypot(center[0] - state.center[0], center[1] - state.center[1])
            if dist < best_dist:
                best_dist = dist
                best_id = track_id

        if best_id is not None and best_dist <= self.track_distance:
            state = self.states[best_id]
            state.center = center
            state.last_seen = now
            return state

        track_id = self.next_track_id
        self.next_track_id += 1
        state = PersonState(track_id=track_id, center=center, last_seen=now)
        self.states[track_id] = state
        return state

    def _update_state_with_pose(self, state: PersonState, pose: Pose, now: float) -> None:
        cfg = self.config
        keypoints = pose.keypoints
        bbox = pose_bbox_from_keypoints(keypoints, cfg.min_kpt_score) or pose.bbox
        ratio, body_w, body_h = bbox_ratio(bbox)
        center_y, center_x = center_from_bbox(bbox)

        angle = torso_angle_degrees(keypoints, cfg.min_kpt_score)
        pair_rule = pair_rule_upper_lower(
            keypoints,
            cfg.min_kpt_score,
            cfg.pair_threshold_y,
            cfg.pair_threshold_x,
        )
        bed_rule = is_on_bed_region(keypoints, cfg.min_kpt_score, cfg.bed_top_y)

        hip = midpoint(keypoints, LEFT_HIP, RIGHT_HIP, cfg.min_kpt_score)
        shoulder = midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, cfg.min_kpt_score)

        hip_speed = 0.0
        shoulder_speed = 0.0
        center_speed = 0.0
        dt = max(1e-3, now - state.prev_time) if state.prev_time is not None else None

        if hip is not None and state.prev_hip_y is not None and dt is not None:
            hip_speed = ((hip[0] - state.prev_hip_y) / dt) / max(body_h, 1e-6)

        if shoulder is not None and state.prev_shoulder_y is not None and dt is not None:
            shoulder_speed = ((shoulder[0] - state.prev_shoulder_y) / dt) / max(body_h, 1e-6)

        if state.prev_center_y is not None and dt is not None:
            center_speed = ((center_y - state.prev_center_y) / dt) / max(body_h, 1e-6)

        max_down_speed = max(hip_speed, shoulder_speed, center_speed)

        soft_motion = max_down_speed >= cfg.soft_drop_speed
        strong_motion = max_down_speed >= cfg.fall_drop_speed
        very_fast_motion = max_down_speed >= cfg.very_fast_drop_speed

        angle_value = -1.0 if angle is None else float(angle)

        upright = (
            angle is not None
            and angle >= cfg.upright_angle
            and ratio <= cfg.upright_max_ratio
        )

        horizontal_by_angle = (
            angle is not None
            and angle <= cfg.horizontal_angle
            and ratio >= cfg.horizontal_ratio
        )

        horizontal_by_pair = (
            pair_rule
            and ratio >= cfg.pair_horizontal_ratio
            and (angle is None or angle <= 45.0)
        )

        geometric_horizontal = horizontal_by_angle or horizontal_by_pair

        if upright:
            state.last_upright_time = now
            state.last_upright_center_y = center_y

            if state.last_status != "FALL":
                state.descent_started_at = None
                state.ground_counter = 0
                state.fall_counter = max(0, state.fall_counter - 1)

        recent_upright = (
            state.last_upright_time is not None
            and now - state.last_upright_time <= cfg.upright_memory_sec
        )

        low_drop = 0.0
        low_enough = False

        if state.last_upright_center_y is not None:
            low_drop = center_y - state.last_upright_center_y
            low_threshold = max(
                cfg.min_low_drop_norm,
                cfg.min_low_drop_body_heights * body_h,
            )
            low_enough = low_drop >= low_threshold

        # Important fix:
        # Some real falls become flat by torso angle, but their bbox ratio stays near 1.0.
        # This happens when the person curls up, knees are close, or keypoints are occluded.
        # So we allow a lower ratio only when the body also dropped from an upright baseline.
        low_horizontal = (
            low_enough
            and angle is not None
            and angle <= cfg.low_horizontal_angle
            and ratio >= cfg.low_horizontal_ratio
        )

        horizontal = geometric_horizontal or low_horizontal

        if strong_motion and not state.was_horizontal:
            state.recent_motion_until = now + cfg.motion_memory_sec

            if recent_upright or (cfg.allow_no_upright_if_very_fast and very_fast_motion):
                state.descent_started_at = now

        recent_motion = now <= state.recent_motion_until

        in_descent_window = (
            state.descent_started_at is not None
            and now - state.descent_started_at <= cfg.descent_timeout_sec
        )

        # Bending rejection:
        # Before, this rejected some true falls.
        # Now it only rejects bending when the body did NOT become low.
        bending_like = (
            shoulder_speed >= cfg.soft_drop_speed
            and hip_speed < cfg.soft_drop_speed * 0.60
            and not horizontal
            and not low_enough
        )

        motion_context = recent_motion or in_descent_window

        has_valid_fall_context = (
            recent_upright
            or low_enough
            or (cfg.allow_no_upright_if_very_fast and very_fast_motion)
        )

        ground_candidate = (
            horizontal
            and has_valid_fall_context
            and motion_context
        )

        # Secondary compact-fall path.
        # This catches the false negatives where status was DESCENDING or POSSIBLE_FALL,
        # motion was true, low was true, but horizontal was false because ratio was too strict.
        compact_fall_candidate = (
            recent_upright
            and low_enough
            and motion_context
            and angle is not None
            and angle <= cfg.low_horizontal_angle
            and ratio >= cfg.low_horizontal_ratio
        )

        ground_candidate = ground_candidate or compact_fall_candidate

        if cfg.allow_static_lying and horizontal and not bed_rule:
            ground_candidate = True

        if bed_rule:
            ground_candidate = False
            horizontal = False

        if ground_candidate:
            state.ground_counter += 1

            high_confidence = (
                low_enough
                or very_fast_motion
                or compact_fall_candidate
            )

            increment = cfg.high_confidence_increment if high_confidence else 1
            state.fall_counter = min(cfg.fall_frames, state.fall_counter + increment)

        else:
            if upright:
                state.ground_counter = 0
            elif not horizontal:
                state.ground_counter = max(0, state.ground_counter - 1)

            state.fall_counter = max(0, state.fall_counter - 1)

        if state.fall_counter >= cfg.fall_frames:
            state.last_status = "FALL"
        elif state.fall_counter > 0:
            state.last_status = "POSSIBLE_FALL"
        elif bending_like:
            state.last_status = "BENDING"
        elif horizontal:
            state.last_status = "LYING"
        elif in_descent_window:
            state.last_status = "DESCENDING"
        else:
            state.last_status = "OK"

        state.was_horizontal = bool(horizontal)

        if hip is not None:
            state.prev_hip_y = hip[0]

        if shoulder is not None:
            state.prev_shoulder_y = shoulder[0]

        state.prev_center_y = center_y
        state.prev_time = now
        state.center = (center_y, center_x)
        state.last_seen = now

        state.debug = {
            "pose_score": float(pose.score),
            "ratio": float(ratio),
            "angle": angle_value,
            "body_w": float(body_w),
            "body_h": float(body_h),
            "center_y": float(center_y),
            "hip_speed": float(hip_speed),
            "shoulder_speed": float(shoulder_speed),
            "center_speed": float(center_speed),
            "max_down_speed": float(max_down_speed),
            "soft_motion": bool(soft_motion),
            "strong_motion": bool(strong_motion),
            "recent_motion": bool(recent_motion),
            "recent_upright": bool(recent_upright),
            "upright": bool(upright),
            "horizontal": bool(horizontal),
            "geometric_horizontal": bool(geometric_horizontal),
            "low_horizontal": bool(low_horizontal),
            "compact_fall_candidate": bool(compact_fall_candidate),
            "low_drop": float(low_drop),
            "low_enough": bool(low_enough),
            "bending_like": bool(bending_like),
            "ground_candidate": bool(ground_candidate),
            "fall_counter": int(state.fall_counter),
            "ground_counter": int(state.ground_counter),
        }
