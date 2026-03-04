import os

# Render (and other PaaS) inject PORT automatically.
# Must bind to 0.0.0.0 so the platform can route external traffic.
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Socket.IO requires eventlet (or gevent) worker.
worker_class = "eventlet"

# Single worker — eventlet handles concurrency internally via green threads.
workers = 1
threads = 1

timeout = 120
graceful_timeout = 30
keepalive = 5

accesslog = "-"
errorlog  = "-"
loglevel  = os.environ.get("LOG_LEVEL", "info")
