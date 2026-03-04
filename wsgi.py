import eventlet
eventlet.monkey_patch()

from app import app, socketio  # noqa: F401 — expose for gunicorn
