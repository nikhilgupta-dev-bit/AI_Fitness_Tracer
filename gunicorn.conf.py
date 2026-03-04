# eventlet MUST be monkey-patched as early as possible.
# Placing it here ensures it runs when gunicorn loads this config.
import eventlet
eventlet.monkey_patch()

import os

# ── Server Socket ────────────────────────────────────────────────────────────
# Render injects PORT automatically; fall back to 10000 (Render's default).
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# ── Workers ──────────────────────────────────────────────────────────────────
# eventlet worker is REQUIRED for Flask-SocketIO WebSocket support.
# Single worker + green threads handles all concurrency via eventlet.
worker_class = "eventlet"
workers      = 1       # DO NOT increase — multiple workers break SocketIO state
threads      = 1       # eventlet uses green threads, not OS threads

# ── Timeouts ─────────────────────────────────────────────────────────────────
# Render's free tier can be slow to cold-start; 120s prevents premature kills.
timeout          = 120
graceful_timeout = 30
keepalive        = 65   # slightly above Render's 60s idle connection timeout

# ── WebSocket / Connection limits ────────────────────────────────────────────
# Raise the number of simultaneous green-thread connections eventlet can hold.
worker_connections = 1000

# ── Logging ──────────────────────────────────────────────────────────────────
# Log to stdout/stderr so Render captures everything in its log dashboard.
accesslog = "-"
errorlog  = "-"
loglevel  = os.environ.get("LOG_LEVEL", "info")

# ── Process naming ──────────────────────
proc_name = "velofit"