# IoT fall detection

The Windows and Raspberry Pi applications share a trained temporal keypoint
classifier. MoveNet or Coral PoseNet supplies 17 normalized keypoints; a compact
random forest evaluates 1.5 seconds of keypoint history. The portable model is
`models/fall_classifier.json` and needs only NumPy at runtime.

## Windows

From the repository root:

```powershell
uv sync
uv run python main_win.py
```

Press `q` or Escape to stop. The Windows evaluator runs the production model:

```powershell
uv run python test_win.py
```

## Raspberry Pi 4 + Coral USB Accelerator

Use Python 3.9 and follow the complete OS-package, virtual-environment, and
Coral setup at the top of `pi_requirements.txt`. Clone Google's PoseNet helper:

```bash
git clone https://github.com/google-coral/project-posenet.git
python3 main_pi.py
```

The application uses the checked-in Edge TPU model under `models/`. Run commands
from the repository root so local modules are importable.

The Pi also exposes:

- `GET http://<pi-ip>:8000/status`
- `GET http://<pi-ip>:8000/metrics`
- `GET http://<pi-ip>:8000/video_feed`
- `GET http://<pi-ip>:8000/stream/status`
- `POST http://<pi-ip>:8000/stream/start`
- `POST http://<pi-ip>:8000/stream/stop`
- `POST http://<pi-ip>:8000/notifications/register`
- `POST http://<pi-ip>:8000/notifications/unregister`
- `GET http://<pi-ip>:8000/notifications/status`
- `WS  ws://<pi-ip>:8000/ws`

See [MOBILE_INTEGRATION.md](MOBILE_INTEGRATION.md) for Flutter setup, event
examples, and the end-to-end demo flow.

Set `DISPLAY = False` in `detectors/pi4_coral_posenet_fall.py` for a headless Pi.

## Firebase push notifications

The Pi sends real system notifications through Firebase Cloud Messaging. Create
a Firebase service-account JSON file, keep it outside source control, and set:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/home/pi/fall-detection/firebase-service-account.json
```

The Flutter app registers its FCM device token with the Pi whenever it connects.
Firebase credentials are optional at process startup: detection and local
WebSocket alerts continue working, while `/notifications/status` reports any
configuration or delivery error.

## Architecture

```text
Windows MoveNet ----\
                     > Pose[] -> keypoint history -> random forest -> alarm/HUD
Pi Coral PoseNet ---/                                      |
                                                           +-> Pi IoT server
```

## Training and held-out test

The GMDCSA24 split is random, stratified, deterministic, and video-level: 75%
training and 25% held out. Reproduce extraction, model selection, and evaluation:

```powershell
uv pip install --python .venv\Scripts\python.exe -r train\requirements.txt
.venv\Scripts\python.exe -m train.extract_keypoints
.venv\Scripts\python.exe -m train.train_classifier
```

Results and the exact split are in `train/report.json`. Only the old detector's
recorded benchmark remains for comparison; its implementation has been removed.
