from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pose_types import Person, bbox_from_keypoints


MOVENET_INPUT_SIZE = 256


def _load_interpreter(model_path: Path):
    try:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=str(model_path))
    except ImportError:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter(model_path=str(model_path))


class MoveNetMultiPoseDetector:
    model_name = "MoveNet MultiPose"

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        self.interpreter = _load_interpreter(model_path)

        input_details = self.interpreter.get_input_details()
        input_index = input_details[0]["index"]
        input_shape = input_details[0]["shape"]

        reported_height = int(input_shape[1])
        reported_width = int(input_shape[2])

        if reported_height < 32 or reported_width < 32:
            self.interpreter.resize_tensor_input(
                input_index,
                [1, MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE, 3],
                strict=False,
            )

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]["shape"]
        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])
        self.input_dtype = self.input_details[0]["dtype"]

        if self.input_height < 32 or self.input_width < 32:
            raise RuntimeError(
                f"Bad MoveNet input shape: {input_shape}. "
                f"Expected something like [1, 256, 256, 3]."
            )

        print(f"MoveNet input shape: {self.input_details[0]['shape']}")
        print(f"MoveNet output shape: {self.output_details[0]['shape']}")

    def detect(self, frame: np.ndarray) -> list[Person]:
        input_image, scale, pad_x, pad_y = self._preprocess(frame)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_image)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        predictions = self._extract_predictions(output)

        frame_height, frame_width = frame.shape[:2]
        people: list[Person] = []

        for row in predictions:
            person_score = float(row[55])
            if person_score <= 0.05:
                continue

            keypoints = np.zeros((17, 3), dtype=np.float32)

            for i in range(17):
                y_norm = float(row[i * 3])
                x_norm = float(row[i * 3 + 1])
                score = float(row[i * 3 + 2])

                x_input = x_norm * self.input_width
                y_input = y_norm * self.input_height

                x = (x_input - pad_x) / scale
                y = (y_input - pad_y) / scale

                keypoints[i] = [
                    np.clip(x, 0, frame_width - 1),
                    np.clip(y, 0, frame_height - 1),
                    score,
                ]

            bbox = bbox_from_keypoints(keypoints, frame_width, frame_height)
            people.append(Person(keypoints=keypoints, score=person_score, bbox=bbox))

        return people

    def _extract_predictions(self, output: np.ndarray) -> np.ndarray:
        output = np.squeeze(output)

        if output.ndim != 2:
            raise RuntimeError(f"Unexpected MoveNet output shape after squeeze: {output.shape}")

        if output.shape[1] == 56:
            return output

        if output.shape[0] == 56:
            return output.T

        raise RuntimeError(
            f"Unexpected MoveNet output shape: {output.shape}. "
            "Expected [6, 56] or similar."
        )

    def _preprocess(self, frame: np.ndarray):
        frame_height, frame_width = frame.shape[:2]

        if frame_height <= 0 or frame_width <= 0:
            raise RuntimeError(f"Bad camera frame shape: {frame.shape}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        scale = min(
            self.input_width / frame_width,
            self.input_height / frame_height,
        )

        resized_width = max(1, int(frame_width * scale))
        resized_height = max(1, int(frame_height * scale))

        resized = cv2.resize(
            rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.zeros(
            (self.input_height, self.input_width, 3),
            dtype=np.uint8,
        )

        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2

        canvas[
            pad_y : pad_y + resized_height,
            pad_x : pad_x + resized_width,
        ] = resized

        input_image = np.expand_dims(canvas, axis=0)

        if self.input_dtype == np.float32:
            input_image = input_image.astype(np.float32)
        elif self.input_dtype == np.int32:
            input_image = input_image.astype(np.int32)
        else:
            input_image = input_image.astype(self.input_dtype)

        return input_image, scale, pad_x, pad_y