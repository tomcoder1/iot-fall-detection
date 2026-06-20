from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

WINDOW_TITLE = "Fall Detection"

# Person filtering
MIN_PERSON_SCORE = 0.20
MIN_KEYPOINT_SCORE = 0.25
MIN_VALID_KEYPOINTS = 6
MIN_BODY_AREA_RATIO = 0.015

# Multi-person safety behavior
MULTI_PERSON_CONFIRM_FRAMES = 2

# Fall geometry
HORIZONTAL_TORSO_DEGREES = 60.0
LYING_BOX_RATIO = 0.85

PAIR_CLOSE_Y_RATIO = 0.16
PAIR_FAR_X_RATIO = 0.20

HIP_DROP_SPEED = 0.30
SHOULDER_DROP_SPEED = 0.30
FALL_DROP_SPEED = 0.75
FALL_MOTION_MEMORY_SECONDS = 1.20

FALL_CONFIRM_FRAMES = 2
LYING_CONFIRM_FRAMES = 3
ALARM_HOLD_SECONDS = 5.0

# Keep False for real detection.
# Slow lying should become LYING, not FALL.
ALLOW_STATIC_LYING = False

# Optional bed/sofa cancellation line.
# Example: 0.55 means upper body below 55 percent of image height
# will not trigger fall. Keep None unless calibrated.
BED_TOP_Y_RATIO = None

# Drawing
DRAW_ALL_PEOPLE = True
DRAW_SKELETON = True
DRAW_HUD = True

# Models
MOVENET_MODEL_PATH = MODELS_DIR / "movenet_multipose_lightning.tflite"
CORAL_POSENET_MODEL_PATH = MODELS_DIR / "posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite"