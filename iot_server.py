import time
import threading
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse


app = FastAPI()

_lock = threading.Lock()
_latest_jpeg: Optional[bytes] = None

_state = {
    "fall": False,
    "status": "STARTING",
    "people": 0,
    "fps": 0.0,
    "timestamp": time.time(),
    "seq": 0,
}


def update_iot_state(
    frame_bgr,
    fall_detected: bool,
    status: str = "OK",
    people: int = 0,
    fps: float = 0.0,
):
    """
    Call this once per camera frame from your fall detection loop.
    """

    global _latest_jpeg, _state

    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 70],
    )

    with _lock:
        if ok:
            _latest_jpeg = encoded.tobytes()

        old_fall = bool(_state["fall"])
        new_fall = bool(fall_detected)

        if old_fall != new_fall or status != _state["status"]:
            _state["seq"] += 1

        _state["fall"] = new_fall
        _state["status"] = status
        _state["people"] = int(people)
        _state["fps"] = float(fps)
        _state["timestamp"] = time.time()


@app.get("/status")
def get_status():
    with _lock:
        return JSONResponse(dict(_state))


def mjpeg_generator():
    while True:
        with _lock:
            frame = _latest_jpeg

        if frame is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

        # Limit stream rate so the Pi is not overloaded.
        time.sleep(0.07)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_status(websocket: WebSocket):
    await websocket.accept()

    last_seq = -1

    try:
        while True:
            with _lock:
                data = dict(_state)

            # Send immediately on state change.
            if data["seq"] != last_seq:
                await websocket.send_json(data)
                last_seq = data["seq"]
            else:
                # Also send heartbeat so the app knows the Pi is alive.
                await websocket.send_json(data)

            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def start_iot_server(host: str = "0.0.0.0", port: int = 8000):
    def run():
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    print(f"IoT server running at http://{host}:{port}")