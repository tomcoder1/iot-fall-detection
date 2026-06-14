import time
from collections import deque

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite_runtime")
except ModuleNotFoundError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("Using tensorflow.lite.Interpreter")

MODEL_PATH = "movenet_multipose_lightning.tflite"

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

INPUT_SIZE = 256
TARGET_FPS = 10.0
NUM_THREADS = 4

KEYPOINT_THRESHOLD = 0.25
PERSON_THRESHOLD = 0.25

SHOW_DEBUG = True
DRAW_ALL_PEOPLE = False

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
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

SKELETON_EDGES = [
    (NOSE, LEFT_EYE),
    (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR),
    (RIGHT_EYE, RIGHT_EAR),
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

def resize_with_pad(frame, target_size):
    """
    Resize the frame while preserving aspect ratio, then pad to square.
    This avoids distorting body shape before MoveNet.
    """
    orig_h, orig_w = frame.shape[:2]

    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    meta = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "target_size": target_size,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }

    return padded, meta


def map_keypoints_to_original(kps, meta):
    """
    MoveNet returns normalized keypoints relative to the padded square input.
    Convert them back to normalized coordinates in the original camera frame.
    """
    mapped = kps.copy()

    target = meta["target_size"]
    scale = meta["scale"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]

    for i in range(mapped.shape[0]):
        y_padded = float(kps[i, 0]) * target
        x_padded = float(kps[i, 1]) * target

        x_orig = (x_padded - pad_x) / scale
        y_orig = (y_padded - pad_y) / scale

        mapped[i, 1] = np.clip(x_orig / orig_w, 0.0, 1.0)
        mapped[i, 0] = np.clip(y_orig / orig_h, 0.0, 1.0)

    return mapped


class MoveNetMultiPose:
    def __init__(self, model_path, input_size=256, num_threads=4):
        self.input_size = input_size

        try:
            self.interpreter = Interpreter(
                model_path=model_path,
                num_threads=num_threads,
            )
        except TypeError:
            self.interpreter = Interpreter(model_path=model_path)

        input_details = self.interpreter.get_input_details()
        input_index = input_details[0]["index"]

        self.interpreter.resize_tensor_input(
            input_index,
            [1, input_size, input_size, 3],
            strict=False,
        )

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_dtype = self.input_details[0]["dtype"]

        print("Model input shape:", self.input_details[0]["shape"])
        print("Model input dtype:", self.input_dtype)
        print("Model output shape:", self.output_details[0]["shape"])

    def detect(self, frame):
        model_input_bgr, meta = resize_with_pad(frame, self.input_size)
        model_input_rgb = cv2.cvtColor(model_input_bgr, cv2.COLOR_BGR2RGB)

        input_data = np.expand_dims(model_input_rgb, axis=0)

        if self.input_dtype == np.float32:
            input_data = input_data.astype(np.float32) / 255.0
        elif self.input_dtype == np.int32:
            input_data = input_data.astype(np.int32)
        else:
            input_data = input_data.astype(self.input_dtype)

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_data,
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        output = np.squeeze(output)

        if output.ndim == 1:
            output = output.reshape(1, -1)

        if output.ndim != 2 or output.shape[1] < 51:
            return []

        people = []

        for row in output:
            kps = row[:51].reshape(17, 3).astype(np.float32)
            kps = map_keypoints_to_original(kps, meta)

            if row.shape[0] >= 56:
                person_score = float(row[55])
            else:
                visible = kps[:, 2] >= KEYPOINT_THRESHOLD
                person_score = float(np.mean(kps[visible, 2])) if np.any(visible) else 0.0

            visible_count = int(np.sum(kps[:, 2] >= KEYPOINT_THRESHOLD))

            if person_score < PERSON_THRESHOLD or visible_count < 5:
                continue

            people.append({
                "keypoints": kps,
                "score": person_score,
                "visible_count": visible_count,
            })

        people.sort(
            key=lambda p: (p["score"], p["visible_count"]),
            reverse=True,
        )

        return people

class RuleBasedFallDetector:
    """
    Rule:
        rapid fall-like motion
        + transition into fallen posture
        + fallen posture persists
        + no quick upright recovery
        = confirmed fall

    This detector uses real elapsed time instead of assuming exact FPS.
    """

    def __init__(self, target_fps=10.0):
        self.target_fps = target_fps

        self.state = "NO_PERSON"
        self.debug = {}
        self.selected_person = None

        self.prev = None
        self.smoothed = None

        self.last_update_time = None
        self.last_rapid_time = -1e9
        self.fall_start_time = None

        self.candidate_time = 0.0
        self.fallen_time = 0.0
        self.still_time = 0.0
        self.upright_time = 0.0
        self.no_person_time = 0.0

        # Smoothing. Higher means faster response, lower means smoother.
        self.smooth_alpha = 0.45

        # Posture thresholds.
        self.upright_angle_max = 35.0
        self.fallen_angle_min = 65.0
        self.fallen_ratio_max = 0.80
        self.combo_fallen_angle_min = 55.0
        self.combo_fallen_ratio_max = 1.05

        # Motion thresholds, normalized coordinates per second.
        self.fast_hip_drop_speed = 0.65
        self.fast_center_drop_speed = 0.60
        self.fast_rotation_speed = 220.0
        self.fast_height_collapse_speed = 0.60

        # Stillness thresholds.
        self.still_center_speed_max = 0.08
        self.still_rotation_speed_max = 35.0

        # Temporal thresholds.
        self.recent_motion_sec = 0.80
        self.candidate_required_sec = 0.15
        self.fallen_required_sec = 0.45
        self.still_required_sec = 0.30
        self.no_recovery_required_sec = 0.90
        self.recovery_required_sec = 1.50
        self.no_person_confirm_sec = 0.30

    def _time_delta(self):
        now = time.time()

        if self.last_update_time is None:
            dt = 1.0 / self.target_fps
        else:
            dt = now - self.last_update_time

        dt = max(0.001, min(dt, 0.5))
        self.last_update_time = now

        return now, dt

    def _reset_fall_timers(self):
        self.candidate_time = 0.0
        self.fallen_time = 0.0
        self.still_time = 0.0
        self.no_person_time = 0.0
        self.fall_start_time = None

    def _point(self, kps, idx):
        y, x, score = kps[idx]

        if score < KEYPOINT_THRESHOLD:
            return None

        return np.array([float(x), float(y)], dtype=np.float32)

    def _midpoint(self, kps, idx_a, idx_b):
        a = self._point(kps, idx_a)
        b = self._point(kps, idx_b)

        if a is not None and b is not None:
            return (a + b) / 2.0

        if a is not None:
            return a

        if b is not None:
            return b

        return None

    def _visible_bbox(self, kps):
        visible = kps[:, 2] >= KEYPOINT_THRESHOLD

        if int(np.sum(visible)) < 5:
            return None

        xs = kps[visible, 1]
        ys = kps[visible, 0]

        x1 = float(np.min(xs))
        y1 = float(np.min(ys))
        x2 = float(np.max(xs))
        y2 = float(np.max(ys))

        width = max(0.001, x2 - x1)
        height = max(0.001, y2 - y1)

        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": width,
            "height": height,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "ratio": height / width,
            "bottom": y2,
            "area": width * height,
        }

    def _torso_angle(self, shoulder_mid, hip_mid):
        if shoulder_mid is None or hip_mid is None:
            return None

        dx = float(hip_mid[0] - shoulder_mid[0])
        dy = float(hip_mid[1] - shoulder_mid[1])

        # 0 degrees means vertical. 90 degrees means horizontal.
        return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))

    def _extract_features(self, person):
        kps = person["keypoints"]
        bbox = self._visible_bbox(kps)

        if bbox is None:
            return None

        shoulder_mid = self._midpoint(kps, LEFT_SHOULDER, RIGHT_SHOULDER)
        hip_mid = self._midpoint(kps, LEFT_HIP, RIGHT_HIP)

        angle = self._torso_angle(shoulder_mid, hip_mid)

        if shoulder_mid is not None and hip_mid is not None:
            center = (shoulder_mid + hip_mid) / 2.0
            center_x = float(center[0])
            center_y = float(center[1])
            hip_y = float(hip_mid[1])
        else:
            center_x = bbox["cx"]
            center_y = bbox["cy"]
            hip_y = bbox["cy"]

        return {
            "center_x": center_x,
            "center_y": center_y,
            "hip_y": hip_y,
            "bbox_width": bbox["width"],
            "bbox_height": bbox["height"],
            "bbox_ratio": bbox["ratio"],
            "bbox_bottom": bbox["bottom"],
            "bbox_area": bbox["area"],
            "torso_angle": angle,
            "person_score": person["score"],
            "visible_count": person["visible_count"],
        }

    def _smooth_features(self, current):
        if self.smoothed is None:
            self.smoothed = current.copy()
            return self.smoothed.copy()

        alpha = self.smooth_alpha
        out = {}

        for key, value in current.items():
            previous = self.smoothed.get(key)

            if isinstance(value, (float, int)) and isinstance(previous, (float, int)):
                out[key] = (1.0 - alpha) * float(previous) + alpha * float(value)
            else:
                out[key] = value

        if current["torso_angle"] is None:
            out["torso_angle"] = self.smoothed.get("torso_angle")
        elif self.smoothed.get("torso_angle") is None:
            out["torso_angle"] = current["torso_angle"]

        self.smoothed = out.copy()
        return out

    def _classify_posture(self, f):
        angle = f["torso_angle"]
        ratio = f["bbox_ratio"]

        fallen_by_angle = angle is not None and angle >= self.fallen_angle_min
        fallen_by_box = ratio <= self.fallen_ratio_max and f["bbox_width"] >= 0.18
        fallen_by_combo = (
            angle is not None
            and angle >= self.combo_fallen_angle_min
            and ratio <= self.combo_fallen_ratio_max
        )

        fallen = fallen_by_angle or fallen_by_box or fallen_by_combo

        upright = (
            angle is not None
            and angle <= self.upright_angle_max
            and ratio >= 1.05
        )

        unstable = not fallen and not upright

        return fallen, upright, unstable

    def _compute_motion(self, current, dt):
        if self.prev is None:
            return {
                "hip_drop_speed": 0.0,
                "center_drop_speed": 0.0,
                "center_speed": 0.0,
                "rotation_speed": 0.0,
                "height_collapse_speed": 0.0,
                "motion_votes": 0,
                "rapid_motion": False,
                "still": False,
            }

        hip_drop = current["hip_y"] - self.prev["hip_y"]
        center_drop = current["center_y"] - self.prev["center_y"]

        dx = current["center_x"] - self.prev["center_x"]
        dy = current["center_y"] - self.prev["center_y"]
        center_distance = float(np.sqrt(dx * dx + dy * dy))

        height_collapse = self.prev["bbox_height"] - current["bbox_height"]

        if current["torso_angle"] is not None and self.prev["torso_angle"] is not None:
            rotation = abs(current["torso_angle"] - self.prev["torso_angle"])
        else:
            rotation = 0.0

        hip_drop_speed = hip_drop / dt
        center_drop_speed = center_drop / dt
        center_speed = center_distance / dt
        rotation_speed = rotation / dt
        height_collapse_speed = height_collapse / dt

        motion_votes = 0

        if hip_drop_speed >= self.fast_hip_drop_speed:
            motion_votes += 1

        if center_drop_speed >= self.fast_center_drop_speed:
            motion_votes += 1

        if rotation_speed >= self.fast_rotation_speed:
            motion_votes += 1

        if height_collapse_speed >= self.fast_height_collapse_speed:
            motion_votes += 1

        rapid_motion = motion_votes >= 2

        if hip_drop_speed >= 1.10:
            rapid_motion = True

        if rotation_speed >= 350.0 and center_drop_speed >= 0.25:
            rapid_motion = True

        still = (
            center_speed <= self.still_center_speed_max
            and rotation_speed <= self.still_rotation_speed_max
        )

        return {
            "hip_drop_speed": hip_drop_speed,
            "center_drop_speed": center_drop_speed,
            "center_speed": center_speed,
            "rotation_speed": rotation_speed,
            "height_collapse_speed": height_collapse_speed,
            "motion_votes": motion_votes,
            "rapid_motion": rapid_motion,
            "still": still,
        }

    def update(self, people):
        now, dt = self._time_delta()

        self.selected_person = people[0] if people else None

        if self.selected_person is None:
            return self._handle_no_person(now, dt)

        self.no_person_time = 0.0

        raw_features = self._extract_features(self.selected_person)

        if raw_features is None:
            return self._handle_no_person(now, dt)

        current = self._smooth_features(raw_features)
        fallen, upright, unstable = self._classify_posture(current)
        motion = self._compute_motion(current, dt)

        if motion["rapid_motion"]:
            self.last_rapid_time = now

        recent_rapid = (now - self.last_rapid_time) <= self.recent_motion_sec

        fall_candidate = fallen and recent_rapid

        if fall_candidate:
            self.candidate_time += dt

            if self.fall_start_time is None:
                self.fall_start_time = now
        else:
            self.candidate_time = max(0.0, self.candidate_time - dt)

        if fallen:
            self.fallen_time += dt
        else:
            self.fallen_time = max(0.0, self.fallen_time - dt)

        if motion["still"]:
            self.still_time += dt
        else:
            self.still_time = 0.0

        if upright:
            self.upright_time += dt
        else:
            self.upright_time = 0.0

        time_since_fall_start = 0.0
        if self.fall_start_time is not None:
            time_since_fall_start = now - self.fall_start_time

        if self.state == "FALL_CONFIRMED":
            if self.upright_time >= self.recovery_required_sec:
                self.state = "NORMAL"
                self._reset_fall_timers()

            self.prev = current
            self._update_debug(current, motion, fallen, upright, unstable, recent_rapid, dt)
            return self.state

        if self.candidate_time >= self.candidate_required_sec:
            self.state = "FALLING"

        if self.state == "FALLING":
            recovered_quickly = self.upright_time >= 0.30

            if recovered_quickly:
                self.state = "NORMAL"
                self._reset_fall_timers()

            else:
                confirmed_by_stillness = (
                    self.fallen_time >= self.fallen_required_sec
                    and self.still_time >= self.still_required_sec
                )

                confirmed_by_no_recovery = (
                    self.fallen_time >= self.fallen_required_sec
                    and time_since_fall_start >= self.no_recovery_required_sec
                    and not upright
                )

                if confirmed_by_stillness or confirmed_by_no_recovery:
                    self.state = "FALL_CONFIRMED"

                elif (
                    not fallen
                    and not recent_rapid
                    and time_since_fall_start >= self.no_recovery_required_sec
                ):
                    self.state = "UNSTABLE"
                    self._reset_fall_timers()

        else:
            if fallen and not recent_rapid:
                self.state = "LYING"

            elif upright:
                self.state = "NORMAL"
                self._reset_fall_timers()

            elif unstable:
                self.state = "UNSTABLE"

            else:
                self.state = "NORMAL"

        self.prev = current
        self._update_debug(current, motion, fallen, upright, unstable, recent_rapid, dt)

        return self.state

    def _handle_no_person(self, now, dt):
        self.no_person_time += dt

        if self.state == "FALL_CONFIRMED":
            return self.state

        recent_rapid = (now - self.last_rapid_time) <= self.recent_motion_sec

        previous_low = (
            self.prev is not None
            and self.prev["bbox_bottom"] >= 0.72
        )

        if self.state == "FALLING":
            if self.no_person_time >= self.no_person_confirm_sec:
                self.state = "FALL_CONFIRMED"
                return self.state

            return "FALLING"

        if (
            recent_rapid
            and previous_low
            and self.no_person_time >= self.no_person_confirm_sec
        ):
            self.state = "FALL_CONFIRMED"
            return self.state

        self.state = "NO_PERSON"
        self.debug = {
            "reason": "no_person",
            "no_person_time": self.no_person_time,
        }

        return self.state

    def _update_debug(self, f, motion, fallen, upright, unstable, recent_rapid, dt):
        angle = f["torso_angle"]

        self.debug = {
            "dt": dt,
            "person_score": f["person_score"],
            "visible_count": f["visible_count"],
            "torso_angle": -1.0 if angle is None else angle,
            "bbox_ratio": f["bbox_ratio"],
            "fallen": fallen,
            "upright": upright,
            "unstable": unstable,
            "recent_rapid": recent_rapid,
            "rapid_motion": motion["rapid_motion"],
            "motion_votes": motion["motion_votes"],
            "still": motion["still"],
            "hip_drop_speed": motion["hip_drop_speed"],
            "center_drop_speed": motion["center_drop_speed"],
            "center_speed": motion["center_speed"],
            "rotation_speed": motion["rotation_speed"],
            "height_collapse_speed": motion["height_collapse_speed"],
            "candidate_time": self.candidate_time,
            "fallen_time": self.fallen_time,
            "still_time": self.still_time,
            "upright_time": self.upright_time,
        }


def draw_person(frame, person, color=(0, 255, 255), kp_threshold=KEYPOINT_THRESHOLD):
    if person is None:
        return

    h, w = frame.shape[:2]
    kps = person["keypoints"]

    for a, b in SKELETON_EDGES:
        if kps[a, 2] >= kp_threshold and kps[b, 2] >= kp_threshold:
            ax = int(kps[a, 1] * w)
            ay = int(kps[a, 0] * h)
            bx = int(kps[b, 1] * w)
            by = int(kps[b, 0] * h)

            cv2.line(frame, (ax, ay), (bx, by), color, 2)

    visible = kps[:, 2] >= kp_threshold

    if np.any(visible):
        xs = kps[visible, 1]
        ys = kps[visible, 0]

        x1 = int(np.min(xs) * w)
        y1 = int(np.min(ys) * h)
        x2 = int(np.max(xs) * w)
        y2 = int(np.max(ys) * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    for kp in kps:
        y, x, score = kp

        if score >= kp_threshold:
            cv2.circle(
                frame,
                (int(x * w), int(y * h)),
                4,
                color,
                -1,
            )


def draw_status(frame, state, fps, debug):
    if state == "FALL_CONFIRMED":
        color = (0, 0, 255)
    elif state == "FALLING":
        color = (0, 165, 255)
    elif state in ("LYING", "UNSTABLE"):
        color = (255, 180, 0)
    elif state == "NO_PERSON":
        color = (180, 180, 180)
    else:
        color = (0, 255, 0)

    cv2.putText(
        frame,
        f"STATE: {state}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if not SHOW_DEBUG:
        return

    keys = [
        "person_score",
        "visible_count",
        "torso_angle",
        "bbox_ratio",
        "fallen",
        "upright",
        "rapid_motion",
        "motion_votes",
        "recent_rapid",
        "still",
        "hip_drop_speed",
        "center_drop_speed",
        "rotation_speed",
        "height_collapse_speed",
        "candidate_time",
        "fallen_time",
        "still_time",
        "upright_time",
    ]

    y = 105

    for key in keys:
        if key not in debug:
            continue

        value = debug[key]

        if isinstance(value, float):
            text = f"{key}: {value:.2f}"
        else:
            text = f"{key}: {value}"

        cv2.putText(
            frame,
            text,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        y += 18

def main():
    model = MoveNetMultiPose(
        model_path=MODEL_PATH,
        input_size=INPUT_SIZE,
        num_threads=NUM_THREADS,
    )

    detector = RuleBasedFallDetector(target_fps=TARGET_FPS)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    target_period = 1.0 / TARGET_FPS
    frame_times = deque(maxlen=30)

    print("Running rule-based fall detection. Press q to quit.")

    while True:
        loop_start = time.time()

        ret, frame = cap.read()

        if not ret:
            print("Camera frame read failed")
            break

        people = model.detect(frame)
        state = detector.update(people)

        if DRAW_ALL_PEOPLE:
            for person in people:
                draw_person(frame, person, color=(80, 180, 80))

        draw_person(frame, detector.selected_person, color=(0, 255, 255))

        frame_times.append(time.time())

        if len(frame_times) >= 2:
            actual_fps = (
                len(frame_times) - 1
            ) / max(1e-6, frame_times[-1] - frame_times[0])
        else:
            actual_fps = 0.0

        # draw_status(frame, state, actual_fps, detector.debug)

        cv2.imshow("Rule-Based Fall Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elapsed = time.time() - loop_start
        sleep_time = target_period - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
