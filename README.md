# IoT fall detection

The Windows and Raspberry Pi applications share the same event-based fall logic
in `detectors/fall_core.py`. Platform modules only perform pose inference; the
camera loop, filtering, drawing, and alarm handling live in `app_common.py`.

## Windows

From the repository root:

```powershell
uv sync
uv run python main_win.py
```

Press `q` or Escape to stop. The Windows evaluator uses the same MoveNet adapter
and fall core:

```powershell
uv run python test_win.py
```

## Raspberry Pi 4 + Coral USB Accelerator

Install OpenCV, Pillow, FastAPI/Uvicorn, and the Coral Edge TPU runtime on the
Pi. Clone Google's PoseNet helper beside this README:

```bash
git clone https://github.com/google-coral/project-posenet.git
python3 main_pi.py
```

The application uses the checked-in Edge TPU model under `models/`. Run commands
from the repository root so local modules are importable.

The Pi also exposes:

- `GET http://<pi-ip>:8000/status`
- `GET http://<pi-ip>:8000/video_feed`
- `WS  ws://<pi-ip>:8000/ws`

Set `DISPLAY = False` in `detectors/pi4_coral_posenet_fall.py` for a headless Pi.

## Architecture

```text
Windows MoveNet ----\
                     > Pose[] -> shared camera runtime -> fall_core -> alarm/HUD
Pi Coral PoseNet ---/                                      |
                                                           +-> Pi IoT server
```

Detection thresholds remain platform-specific `CONFIG` objects so the existing
dataset evaluator and deployed Coral configuration retain their current values.
