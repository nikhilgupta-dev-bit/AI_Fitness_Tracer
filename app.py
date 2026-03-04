import eventlet
eventlet.monkey_patch()

import base64
import os
import atexit

from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit

from pose_engine import FrameProcessor
from rep_counter_lib import EXERCISES

# ── VeloFit AI Setup ─────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "prod-secret-key-123")

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    logger=False,
    engineio_logger=False,
    max_http_buffer_size=5 * 1024 * 1024
)

processor = FrameProcessor(
    model_complexity=int(os.environ.get("MODEL_COMPLEXITY", "0"))
)

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})

@app.get("/api/config")
def api_config():
    return jsonify({"exercises": list(EXERCISES.keys())})

@socketio.on("connect")
def on_connect():
    emit("connected", {"exercises": list(EXERCISES.keys())})

@socketio.on("frame")
def handle_frame(data):
    try:
        jpeg = base64.b64decode(data.get("frame", ""))
        exercise = data.get("exercise", "")
        result = processor.process(jpeg, exercise)
        emit("result", result)
    except Exception as exc:
        emit("error", {"message": str(exc)})

@socketio.on("reset")
def handle_reset():
    processor.reset()
    emit("reset_ok", {})

@socketio.on("set_exercise")
def handle_set_exercise(data):
    exercise = data.get("exercise", "")
    processor.set_exercise(exercise)

@atexit.register
def _shutdown():
    try:
        processor.close()
    except Exception:
        pass

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    socketio.run(app, host=host, port=port, debug=False)