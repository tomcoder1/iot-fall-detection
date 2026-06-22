from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterable, List

try:
    import firebase_admin
    from firebase_admin import credentials, messaging
except ImportError:  # The detector remains usable before FCM is configured.
    firebase_admin = None
    credentials = None
    messaging = None

TOKEN_FILE = Path(
    os.environ.get(
        "FCM_TOKEN_FILE",
        str(Path(__file__).resolve().with_name("fcm_tokens.json")),
    )
)

_lock = threading.Lock()
_tokens = set()
_firebase_ready = False
_last_error = None

def _load_tokens() -> None:
    global _tokens, _last_error
    try:
        values = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        if isinstance(values, list):
            _tokens = {str(value).strip() for value in values if str(value).strip()}
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError) as exc:
        _last_error = f"Could not read FCM token file: {exc}"

def _save_tokens_unlocked() -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = TOKEN_FILE.with_suffix(TOKEN_FILE.suffix + ".tmp")
    temporary.write_text(
        json.dumps(sorted(_tokens), indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, TOKEN_FILE)

def register_token(token: str) -> int:
    token = token.strip()
    if not token:
        raise ValueError("FCM token must not be empty")
    with _lock:
        _tokens.add(token)
        _save_tokens_unlocked()
        return len(_tokens)

def unregister_token(token: str) -> int:
    with _lock:
        _tokens.discard(token.strip())
        _save_tokens_unlocked()
        return len(_tokens)

def _ensure_firebase() -> bool:
    global _firebase_ready, _last_error
    if _firebase_ready:
        return True
    if firebase_admin is None or credentials is None:
        _last_error = "firebase-admin is not installed"
        return False
    try:
        try:
            firebase_admin.get_app()
        except ValueError:
            credential_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not credential_path:
                _last_error = "GOOGLE_APPLICATION_CREDENTIALS is not set"
                return False
            if not Path(credential_path).is_file():
                _last_error = "Firebase service-account file does not exist"
                return False
            firebase_admin.initialize_app(credentials.Certificate(credential_path))
        _firebase_ready = True
        _last_error = None
        return True
    except Exception as exc:
        _last_error = f"Firebase initialization failed: {exc}"
        return False

def notification_status() -> Dict[str, object]:
    _ensure_firebase()
    with _lock:
        count = len(_tokens)
        error = _last_error
    return {
        "configured": _firebase_ready,
        "registered_devices": count,
        "last_error": error,
    }

def _chunks(values: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]

def send_fall_alert(alert: Dict[str, str]) -> None:
    global _last_error
    if not _ensure_firebase() or messaging is None:
        return

    with _lock:
        tokens = sorted(_tokens)
    if not tokens:
        return

    invalid_tokens = set()
    try:
        for token_chunk in _chunks(tokens, 500):
            message = messaging.MulticastMessage(
                tokens=token_chunk,
                notification=messaging.Notification(
                    title="Fall alert",
                    body=alert["message"],
                ),
                data={key: str(value) for key, value in alert.items()},
                android=messaging.AndroidConfig(
                    priority="high",
                    notification=messaging.AndroidNotification(
                        channel_id="fall_alerts",
                        sound="default",
                    ),
                ),
                apns=messaging.APNSConfig(
                    headers={"apns-priority": "10"},
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(sound="default")
                    ),
                ),
            )
            response = messaging.send_each_for_multicast(message)
            for token, result in zip(token_chunk, response.responses):
                if result.success:
                    continue
                error_name = type(result.exception).__name__
                if error_name in {"UnregisteredError", "InvalidArgumentError"}:
                    invalid_tokens.add(token)

        if invalid_tokens:
            with _lock:
                _tokens.difference_update(invalid_tokens)
                _save_tokens_unlocked()
        _last_error = None
    except Exception as exc:
        _last_error = f"FCM send failed: {exc}"

def queue_fall_alert(alert: Dict[str, str]) -> None:
    threading.Thread(
        target=send_fall_alert,
        args=(dict(alert),),
        name="fcm-fall-alert",
        daemon=True,
    ).start()

_load_tokens()