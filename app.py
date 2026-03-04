import atexit
import os
import time
from typing import Generator

from flask import Flask, Response, jsonify, render_template, request

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig")
)
os.environ.setdefault("MPLBACKEND", "Agg")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from pose_engine import PoseRepEngine
from rep_counter_lib import EXERCISES


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


ENGINE = PoseRepEngine(
    camera_index=_env_int("REPTRACK_CAMERA_INDEX", 0),
    frame_size=(
        _env_int("REPTRACK_FRAME_WIDTH", 960),
        _env_int("REPTRACK_FRAME_HEIGHT", 540),
    ),
    process_size=(
        _env_int("REPTRACK_PROCESS_WIDTH", 640),
        _env_int("REPTRACK_PROCESS_HEIGHT", 360),
    ),
    stream_size=(
        _env_int("REPTRACK_STREAM_WIDTH", 960),
        _env_int("REPTRACK_STREAM_HEIGHT", 540),
    ),
    model_complexity=_env_int("REPTRACK_MODEL_COMPLEXITY", 0),
    jpeg_quality=_env_int("REPTRACK_JPEG_QUALITY", 70),
    draw_landmarks=_env_bool("REPTRACK_DRAW_LANDMARKS", True),
    process_every_n=_env_int("REPTRACK_PROCESS_EVERY_N", 1),
)


def _security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    resp.headers["Cache-Control"] = "no-store"
    return resp


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("REPTRACK_SECRET_KEY", "dev-not-for-production")

    @app.after_request
    def _after(resp):
        return _security_headers(resp)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.get("/api/config")
    def api_config():
        return jsonify({"exercises": list(EXERCISES.keys())})

    @app.get("/api/session/state")
    def api_state():
        return Response(ENGINE.get_snapshot_json(), mimetype="application/json")

    @app.post("/api/session/start")
    def api_start():
        data = request.get_json(silent=True) or {}
        exercise = data.get("exercise") or list(EXERCISES.keys())[0]
        ENGINE.start(exercise_name=exercise)
        return Response(ENGINE.get_snapshot_json(), mimetype="application/json")

    @app.post("/api/session/reset")
    def api_reset():
        ENGINE.reset()
        return Response(ENGINE.get_snapshot_json(), mimetype="application/json")

    @app.post("/api/session/exercise")
    def api_exercise():
        data = request.get_json(silent=True) or {}
        exercise = data.get("exercise") or list(EXERCISES.keys())[0]
        ENGINE.set_exercise(exercise)
        return Response(ENGINE.get_snapshot_json(), mimetype="application/json")

    @app.post("/api/session/stop")
    def api_stop():
        ENGINE.stop()
        return jsonify({"ok": True})

    @app.get("/api/session/events")
    def api_events():
        def gen() -> Generator[str, None, None]:
            while True:
                yield f"data: {ENGINE.get_snapshot_json()}\n\n"
                time.sleep(0.12)

        return Response(
            gen(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/video")
    def api_video():
        boundary = "frame"

        def gen() -> Generator[bytes, None, None]:
            while True:
                jpeg = ENGINE.get_jpeg()
                if jpeg:
                    yield (
                        b"--" + boundary.encode("utf-8") + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode("utf-8") + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                else:
                    time.sleep(0.05)

        return Response(
            gen(),
            mimetype=f"multipart/x-mixed-replace; boundary={boundary}",
            headers={"Cache-Control": "no-cache"},
        )

    return app


app = create_app()


@atexit.register
def _shutdown():
    try:
        ENGINE.stop()
    except Exception:
        pass


if __name__ == "__main__":
    host = os.environ.get("REPTRACK_HOST", "127.0.0.1")
    port = int(os.environ.get("REPTRACK_PORT", "8000"))
    debug = os.environ.get("REPTRACK_DEBUG", "0") == "1"
    # Prime camera authorization on the main thread (macOS requirement).
    try:
        print(ENGINE.prime_camera_access(), flush=True)
    except Exception as e:
        print(f"prime_error: {e}", flush=True)

    app.run(host=host, port=port, debug=debug, threaded=True)