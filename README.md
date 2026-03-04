# Pose Detection Rep Counter

Simple webcam app that detects your body pose and counts reps for **bicep curls** using elbow angle.

## Setup

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

## Run (Web UI)

```bash
python app.py
```

Then open `http://127.0.0.1:8000` and press **Start**.

Notes:
- The Web UI uses the **server machine's webcam** (works great when running locally).
- On macOS, you must grant **Camera permission** to the app running Python (usually **Cursor**):  
  System Settings → Privacy & Security → Camera → enable Cursor.
- Configure with env vars:
  - `REPTRACK_CAMERA_INDEX` (default `0`)
  - `REPTRACK_HOST` (default `127.0.0.1`)
  - `REPTRACK_PORT` (default `8000`)

## Run (Production)

```bash
gunicorn -c gunicorn.conf.py wsgi:app
```

Keep `REPTRACK_WORKERS=1` (camera capture is not multi-process friendly).

## Performance tuning (if counting feels slow)

Set these env vars before starting:
- `REPTRACK_MODEL_COMPLEXITY=0` (fastest) or `1` (more accurate, slower)
- `REPTRACK_PROCESS_WIDTH=640` `REPTRACK_PROCESS_HEIGHT=360` (lower = faster)
- `REPTRACK_STREAM_WIDTH=960` `REPTRACK_STREAM_HEIGHT=540` (lower = faster MJPEG)
- `REPTRACK_JPEG_QUALITY=60` (lower = faster streaming)
- `REPTRACK_PROCESS_EVERY_N=2` (process every 2nd frame, faster)
- `REPTRACK_DRAW_LANDMARKS=0` (disable skeleton drawing, faster)

## Run (CLI / OpenCV window)

```bash
python rep_counter_cli.py
```

- Press `q` to quit.
- Press `s` for summary.
- Press `n` to switch exercises.
- Press `r` to reset.

## How counting works

- Each exercise has a configured 3-joint angle (e.g. shoulder–elbow–wrist, hip–knee–ankle).
- The app computes that joint angle each frame, smooths it, and applies hysteresis + hold frames + cooldown.
- A rep is counted on a validated stage transition (start → end).

## Shadow Boxing Mode (OpenCV)

You can run a separate shadow-boxing tracker that detects straight punches and combo bursts.

```bash
python shadow_boxing_cv.py
```

Optional flags:

```bash
python shadow_boxing_cv.py --camera 0 --min-detect 0.6 --min-track 0.6
```

Controls:
- `q` quit
- `r` reset stats

Live metrics shown:
- Total punches
- Left vs right punches
- Current combo and best combo
- Punches per minute (PPM)
