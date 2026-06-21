from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from detectors.fall_core import (
    FallConfig,
    FallDetector,
    PersonState,
    Pose,
    SKELETON_EDGES,
    pose_bbox_from_keypoints,
)


class PoseModel(Protocol):
    """Platform-specific pose model consumed by the shared camera loop."""

    def infer(self, frame_bgr: np.ndarray) -> Tuple[List[Pose], Dict[str, object]]:
        ...


StateSink = Callable[[np.ndarray, bool, str, int, float, Optional[str]], None]


@dataclass(frozen=True)
class AppOptions:
    title: str
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    display: bool = True
    draw_pose: bool = True
    mirror_image: bool = False
    debug_every_n_frames: int = 30
    camera_backend: Optional[int] = None


def run_app(
    pose_model: PoseModel,
    config: FallConfig,
    options: AppOptions,
    state_sink: Optional[StateSink] = None,
) -> int:
    """Run the shared fall-detection loop without changing fall-core behavior."""

    detector = FallDetector(config)
    cap = _open_camera(options)

    frame_idx = 0
    processed_frames = 0
    started_at = time.monotonic()
    fall_alarm_until = 0.0
    multi_person_hits = 0

    print(f"[INFO] Started {options.title}.")
    print(
        f"[INFO] Camera: {options.camera_width}x{options.camera_height} "
        f"at {options.camera_fps} FPS"
    )
    print("[INFO] Rule: if 2+ people are visible, fall detection is OFF.")
    if options.display:
        print("[INFO] Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[INFO] Camera read failed.")
                break

            frame_idx += 1
            if options.mirror_image:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            poses, model_info = pose_model.infer(frame)
            accepted = detector.accepted_poses(poses)
            people_count = len(accepted)

            if people_count > 1:
                multi_person_hits += 1
            else:
                multi_person_hits = 0

            multi_person_disabled = (
                config.stop_when_multiple_people
                and multi_person_hits >= config.multi_person_confirm_frames
            )
            disabled_reason: Optional[str] = None
            results: List[Tuple[Pose, PersonState]] = []

            # These branches intentionally match the previously validated loops.
            if people_count == 0:
                detector.reset()
                fall_alarm_until = 0.0
                fall_detected = False
                disabled_reason = "NO PERSON"
            elif multi_person_disabled:
                detector.reset()
                fall_alarm_until = 0.0
                fall_detected = False
                disabled_reason = "MULTI-PERSON: DETECTION STOPPED"
            else:
                results = detector.update(accepted[:1], now)
                current_fall = any(
                    state.last_status == "FALL" for state in detector.states.values()
                )
                if current_fall:
                    fall_alarm_until = now + config.alarm_hold_sec
                fall_detected = now <= fall_alarm_until

            processed_frames += 1
            elapsed = max(1e-6, time.monotonic() - started_at)
            fps = processed_frames / elapsed

            if options.draw_pose:
                _draw_results(frame, accepted, results, multi_person_disabled, config)
            draw_hud(
                frame,
                fall_detected,
                fps,
                people_count,
                model_info,
                disabled_reason,
            )

            status = disabled_reason or ("FALL" if fall_detected else "OK")
            if state_sink is not None:
                state_sink(
                    frame,
                    fall_detected,
                    status,
                    people_count,
                    fps,
                    disabled_reason,
                )

            if options.debug_every_n_frames and frame_idx % options.debug_every_n_frames == 0:
                debug_states = [state.debug for _, state in results]
                print(
                    f"[DEBUG] frame={frame_idx} fps={fps:.1f} people={people_count} "
                    f"fall_detected={fall_detected} states={debug_states}"
                )

            if options.display:
                cv2.imshow(options.title, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if options.display:
            cv2.destroyAllWindows()

    print("[INFO] Done.")
    return 0


def _open_camera(options: AppOptions) -> cv2.VideoCapture:
    if options.camera_backend is None:
        cap = cv2.VideoCapture(options.camera_index)
    else:
        cap = cv2.VideoCapture(options.camera_index, options.camera_backend)

    _configure_camera(cap, options)

    # DirectShow can be unavailable for a particular Windows camera/driver.
    if not cap.isOpened() and options.camera_backend is not None:
        cap.release()
        cap = cv2.VideoCapture(options.camera_index)
        _configure_camera(cap, options)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {options.camera_index}")
    return cap


def _configure_camera(cap: cv2.VideoCapture, options: AppOptions) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, options.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, options.camera_height)
    cap.set(cv2.CAP_PROP_FPS, options.camera_fps)


def _draw_results(
    frame: np.ndarray,
    accepted: List[Pose],
    results: List[Tuple[Pose, PersonState]],
    multi_person_disabled: bool,
    config: FallConfig,
) -> None:
    if multi_person_disabled:
        for pose in accepted:
            draw_pose(frame, pose, None, config.min_kpt_score, disabled=True)
        return

    result_pose_ids = {id(pose) for pose, _ in results}
    for pose, state in results:
        draw_pose(frame, pose, state, config.min_kpt_score)
    for pose in accepted:
        if id(pose) not in result_pose_ids:
            draw_pose(frame, pose, None, config.min_kpt_score, disabled=True)


def draw_pose(
    frame: np.ndarray,
    pose: Pose,
    state: Optional[PersonState],
    min_kpt_score: float,
    disabled: bool = False,
) -> None:
    h, w = frame.shape[:2]
    keypoints = pose.keypoints

    if disabled:
        color = (0, 255, 255)
    elif state is not None and state.last_status == "FALL":
        color = (0, 0, 255)
    elif state is not None and state.last_status == "LYING":
        color = (255, 180, 0)
    elif state is not None and state.last_status in {"POSSIBLE_FALL", "DESCENDING"}:
        color = (0, 165, 255)
    elif state is not None and state.last_status == "BENDING":
        color = (255, 255, 0)
    else:
        color = (0, 200, 0)

    for a, b in SKELETON_EDGES:
        ya, xa, score_a = keypoints[a]
        yb, xb, score_b = keypoints[b]
        if score_a >= min_kpt_score and score_b >= min_kpt_score:
            cv2.line(
                frame,
                (int(xa * w), int(ya * h)),
                (int(xb * w), int(yb * h)),
                color,
                2,
            )

    for y, x, score in keypoints:
        if score >= min_kpt_score:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 255), -1)

    bbox = pose_bbox_from_keypoints(keypoints, min_kpt_score) or pose.bbox
    ymin, xmin, ymax, xmax = bbox
    cv2.rectangle(
        frame,
        (int(xmin * w), int(ymin * h)),
        (int(xmax * w), int(ymax * h)),
        color,
        2,
    )

    if disabled:
        label = "MULTI-PERSON: FALL OFF"
    elif state is None:
        label = f"score={pose.score:.2f}"
    else:
        debug = state.debug
        label = (
            f"id={state.track_id} {state.last_status} "
            f"cnt={debug.get('fall_counter', 0)} "
            f"r={float(debug.get('ratio', 0)):.2f} "
            f"a={float(debug.get('angle', -1)):.0f} "
            f"v={float(debug.get('max_down_speed', 0)):.2f}"
        )
    cv2.putText(
        frame,
        label,
        (int(xmin * w), max(20, int(ymin * h) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def draw_hud(
    frame: np.ndarray,
    fall_detected: bool,
    fps: float,
    people_count: int,
    model_info: Dict[str, object],
    disabled_reason: Optional[str],
) -> None:
    if disabled_reason:
        status_text = f"fall_detected = False | {disabled_reason}"
        status_color = (0, 255, 255)
    else:
        status_text = f"fall_detected = {fall_detected}"
        status_color = (0, 0, 255) if fall_detected else (0, 200, 0)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_color, 2)
    cv2.putText(
        frame,
        f"FPS(avg): {fps:.1f} | accepted people: {people_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"model: {model_info.get('model_type', '?')} | "
        f"infer: {float(model_info.get('inference_ms', 0.0)):.1f} ms",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
    )
