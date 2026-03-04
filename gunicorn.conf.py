import os

bind = os.environ.get("REPTRACK_BIND", "127.0.0.1:8000")

# OpenCV + camera capture is not multi-process friendly.
workers = int(os.environ.get("REPTRACK_WORKERS", "1"))
threads = int(os.environ.get("REPTRACK_THREADS", "1"))

timeout = int(os.environ.get("REPTRACK_TIMEOUT", "0"))  # 0 = disabled
graceful_timeout = int(os.environ.get("REPTRACK_GRACEFUL_TIMEOUT", "30"))

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("REPTRACK_LOGLEVEL", "info")

