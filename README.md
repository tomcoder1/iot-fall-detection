# IoT fall detection

Windows and Raspberry Pi use separate temporal keypoint classifiers because
MoveNet and Coral PoseNet produce different keypoint distributions. Each compact
random forest evaluates 1.5 seconds of history and needs only NumPy at runtime:

- `models/fall_classifier_windows.json`
- `models/fall_classifier_pi.json`

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
Windows MoveNet -> Windows forest --\
                                   > alarm/HUD
Pi Coral PoseNet -> Pi forest -----/     |
                                         +-> Pi IoT server
```

## Training and held-out test

The GMDCSA24 split is random, stratified, deterministic, and video-level: 75%
training and 25% held out. Reproduce extraction, model selection, and evaluation:

Windows extraction and training:

```powershell
uv pip install --python .venv\Scripts\python.exe -r train\requirements.txt
.venv\Scripts\python.exe -m train.extract_keypoints --platform windows
.venv\Scripts\python.exe -m train.train_classifier --platform windows
```

Pi keypoints must be extracted on the Pi, then the cache is trained on Windows:

```bash
python -m train.extract_keypoints --platform pi
```

```powershell
.venv\Scripts\python.exe -m train.train_classifier --platform pi
```

Exact splits and metrics are in `train/report_windows.json` and
`train/report_pi.json`.

`test_win.py` and `test_pi.py` test only the held-out 25% by default. Pass
`--all` when you intentionally want to include training videos.
