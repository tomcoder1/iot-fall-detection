import importlib
import cv2

REQUIRED_PACKAGES = {
    "OpenCV": "cv2",
    "NumPy": "numpy",
    "Picamera2": "picamera2",
}

print("Fall Detection Setup Test")
print("-------------------------")

for name, module in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(module)
        print(f"[OK] {name} is installed")
    except ImportError:
        print(f"[MISSING] {name} is not installed")

print("\nChecking camera...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("[ERROR] Camera not detected")
else:
    ret, frame = camera.read()
    if ret:
        print("[OK] Camera detected and frame captured")
        print(f"Frame size: {frame.shape[1]} x {frame.shape[0]}")
    else:
        print("[ERROR] Camera detected, but no frame captured")

camera.release()
print("\nTest complete.")
