from __future__ import annotations

import asyncio
import shutil
import subprocess
import threading
import time
from datetime import datetime
from typing import Dict, Optional

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from push_notifications import (
    notification_status,
    queue_fall_alert,
    register_token,
    unregister_token,
)

try:
    import psutil  # type: ignore
except ImportError:  # Optional on a minimal Raspberry Pi install.
    psutil = None


ALERT_MESSAGE = "possible fall detected"
app = FastAPI(title="Fall Detection IoT API")

_lock = threading.Lock()
_started_at = time.monotonic()
_latest_frame = None
_latest_frame_monotonic: Optional[float] = None
_frame_sequence = 0
_active_stream_clients = 0
_stream_requested = False
_previous_fall = False

_state: Dict[str, object] = {
    "fall_detected": False,
    "message": "normal",
    "people_count": 0,
    "disabled_reason": "STARTING",
    "latest_alert": None,
    "detection_fps": 0.0,
    "last_alert_time": None,
    "alert_count": 0,
}


class NotificationToken(BaseModel):
    token: str = Field(min_length=1, max_length=4096)


def _stream_active_unlocked() -> bool:
    return _stream_requested or _active_stream_clients > 0


def _status_unlocked() -> Dict[str, object]:
    return {
        "fall_detected": _state["fall_detected"],
        "message": _state["message"],
        "people_count": _state["people_count"],
        "disabled_reason": _state["disabled_reason"],
        "latest_alert": _state["latest_alert"],
        "stream_active": _stream_active_unlocked(),
    }


def _status_event_unlocked() -> Dict[str, object]:
    status = _status_unlocked()
    status.pop("latest_alert")
    return {"type": "status_update", **status}


def update_iot_state(
    frame_bgr,
    fall_detected: bool,
    status: str = "OK",
    people: int = 0,
    fps: float = 0.0,
    disabled_reason: Optional[str] = None,
) -> None:
    """Receive one raw camera frame and the latest detection state."""

    global _latest_frame, _latest_frame_monotonic, _frame_sequence
    global _previous_fall

    now_monotonic = time.monotonic()
    new_fall = bool(fall_detected)
    push_alert = None

    with _lock:
        # Keep a raw copy. JPEG encoding happens only in an active stream client.
        _latest_frame = frame_bgr.copy()
        _latest_frame_monotonic = now_monotonic
        _frame_sequence += 1

        rising_edge = new_fall and not _previous_fall
        if rising_edge:
            timestamp = datetime.now().astimezone()
            event_id = timestamp.strftime("%Y%m%d-%H%M%S")
            iso_timestamp = timestamp.isoformat(timespec="seconds")
            alert = {
                "event_id": event_id,
                "timestamp": iso_timestamp,
                "message": ALERT_MESSAGE,
                "status": "possible_fall",
                "stream_url": "/video_feed",
            }
            _state["latest_alert"] = alert
            _state["last_alert_time"] = iso_timestamp
            _state["alert_count"] = int(_state["alert_count"]) + 1
            push_alert = alert

        _previous_fall = new_fall
        _state["fall_detected"] = new_fall
        _state["message"] = ALERT_MESSAGE if new_fall else "normal"
        _state["people_count"] = int(people)
        _state["disabled_reason"] = disabled_reason
        _state["detection_fps"] = float(fps)

    if push_alert is not None:
        queue_fall_alert(push_alert)


@app.get("/status")
def get_status():
    with _lock:
        return JSONResponse(_status_unlocked())


@app.get("/notifications/status")
def get_notification_status():
    return JSONResponse(notification_status())


@app.post("/notifications/register")
def register_notification_device(request: NotificationToken):
    device_count = register_token(request.token)
    return JSONResponse(
        {
            "registered": True,
            "registered_devices": device_count,
            "delivery": notification_status(),
        }
    )


@app.post("/notifications/unregister")
def unregister_notification_device(request: NotificationToken):
    device_count = unregister_token(request.token)
    return JSONResponse({"registered": False, "registered_devices": device_count})


def _stream_details_unlocked() -> Dict[str, object]:
    age = None
    if _latest_frame_monotonic is not None:
        age = round(max(0.0, time.monotonic() - _latest_frame_monotonic), 3)
    return {
        "stream_active": _stream_active_unlocked(),
        "active_stream_clients": _active_stream_clients,
        "last_frame_age_sec": age,
        "alert_mode": "possible_fall" if _state["fall_detected"] else "normal",
    }


@app.get("/stream/status")
def stream_status():
    with _lock:
        return JSONResponse(_stream_details_unlocked())


@app.post("/stream/start")
def stream_start():
    global _stream_requested
    with _lock:
        _stream_requested = True
        return JSONResponse(_stream_details_unlocked())


@app.post("/stream/stop")
def stream_stop():
    global _stream_requested
    with _lock:
        _stream_requested = False
        return JSONResponse(_stream_details_unlocked())


def mjpeg_generator():
    global _active_stream_clients
    with _lock:
        _active_stream_clients += 1

    last_sequence = -1
    try:
        while True:
            with _lock:
                sequence = _frame_sequence
                frame = None if _latest_frame is None else _latest_frame.copy()

            if frame is None or sequence == last_sequence:
                time.sleep(0.03)
                continue

            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )
            if not ok:
                time.sleep(0.03)
                continue

            last_sequence = sequence
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )
    finally:
        with _lock:
            _active_stream_clients = max(0, _active_stream_clients - 1)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _cpu_temperature_c() -> Optional[float]:
    command = shutil.which("vcgencmd")
    if command is None:
        return None
    try:
        output = subprocess.run(
            [command, "measure_temp"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=True,
        ).stdout.strip()
        return float(output.split("=")[1].split("'")[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


@app.get("/metrics")
def metrics():
    with _lock:
        result: Dict[str, object] = {
            "detection_fps": _state["detection_fps"],
            "stream_active": _stream_active_unlocked(),
            "active_stream_clients": _active_stream_clients,
            "people_count": _state["people_count"],
            "fall_detected": _state["fall_detected"],
            "last_alert_time": _state["last_alert_time"],
            "alert_count": _state["alert_count"],
            "uptime_sec": round(time.monotonic() - _started_at, 1),
            "cpu_percent": None,
            "memory_percent": None,
            "cpu_temperature_c": None,
        }

    if psutil is not None:
        try:
            result["cpu_percent"] = psutil.cpu_percent(interval=None)
            result["memory_percent"] = psutil.virtual_memory().percent
        except Exception:
            pass
    result["cpu_temperature_c"] = _cpu_temperature_c()
    return JSONResponse(result)


@app.websocket("/ws")
async def websocket_status(websocket: WebSocket):
    await websocket.accept()
    with _lock:
        existing_alert = _state["latest_alert"]
        last_alert_id = (
            existing_alert.get("event_id")
            if isinstance(existing_alert, dict) and not _state["fall_detected"]
            else None
        )
    try:
        while True:
            with _lock:
                status_event = _status_event_unlocked()
                latest_alert = _state["latest_alert"]
                alert_event = dict(latest_alert) if isinstance(latest_alert, dict) else None

            await websocket.send_json(status_event)
            if alert_event is not None and alert_event["event_id"] != last_alert_id:
                await websocket.send_json({"type": "fall_alert", **alert_event})
                last_alert_id = alert_event["event_id"]
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def start_iot_server(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    def run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    print(f"IoT server running at http://{host}:{port}")
    return thread
