import json
import threading
import time
from dataclasses import asdict
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import sys
import os

from rep_counter_lib import EXERCISES, RepCounter, RepSnapshot


class PoseRepEngine:
    """
    Captures frames from a local camera, runs MediaPipe Pose, updates RepCounter,
    and exposes the latest JPEG + snapshot in a thread-safe way.
    """

    def __init__(
        self,
        camera_index: int = 0,
        frame_size: Tuple[int, int] = (960, 540),
        process_size: Tuple[int, int] = (640, 360),
        stream_size: Optional[Tuple[int, int]] = None,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.65,
        jpeg_quality: int = 80,
        draw_landmarks: bool = True,
        process_every_n: int = 1,
    ):
        # OpenCV AVFoundation camera permission prompts must happen on the main
        # thread; our capture loop runs in a background thread.
        if sys.platform == "darwin":
            os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

        self._camera_index = int(camera_index)
        self._frame_size = (int(frame_size[0]), int(frame_size[1]))
        self._process_size = (int(process_size[0]), int(process_size[1]))
        self._stream_size = (
            (int(stream_size[0]), int(stream_size[1])) if stream_size else None
        )
        self._model_complexity = int(model_complexity)
        self._min_det = float(min_detection_confidence)
        self._min_track = float(min_tracking_confidence)
        self._jpeg_quality = int(jpeg_quality)
        self._draw_landmarks = bool(draw_landmarks)
        self._process_every_n = max(1, int(process_every_n))

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._counter = RepCounter(list(EXERCISES.keys())[0])
        self._latest_jpeg: Optional[bytes] = None
        self._latest_snapshot: RepSnapshot = self._counter.snapshot(fps=0.0)
        self._running = False

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._running)

    def start(self, exercise_name: Optional[str] = None) -> None:
        with self._lock:
            if exercise_name:
                self._counter = RepCounter(exercise_name)
            else:
                self._counter.reset()
            self._latest_snapshot = self._counter.snapshot(fps=0.0)
            self._latest_jpeg = None

        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pose-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        with self._lock:
            self._running = False

    def reset(self) -> None:
        with self._lock:
            self._counter.reset()
            self._latest_snapshot = self._counter.snapshot(fps=0.0)

    def set_exercise(self, exercise_name: str) -> None:
        with self._lock:
            self._counter = RepCounter(exercise_name)
            self._latest_snapshot = self._counter.snapshot(fps=0.0)

    def get_snapshot(self) -> RepSnapshot:
        with self._lock:
            return self._latest_snapshot

    def get_snapshot_json(self) -> str:
        snap = self.get_snapshot()
        payload = asdict(snap)
        payload.update(
            {
                "elapsed_sec": snap.elapsed_sec(),
                "rpm": snap.reps_per_min(),
                "avg_cadence": snap.avg_cadence(),
                "avg_rom": snap.avg_rom(),
                "running": self.is_running(),
            }
        )
        return json.dumps(payload, separators=(",", ":"))

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def prime_camera_access(self) -> str:
        """
        Attempts to open the camera on the *current thread* to ensure macOS
        camera authorization has already been granted before the background
        capture thread starts.
        """
        cap, note = self._try_open_camera()
        if not cap:
            return f"prime_failed: {note}"
        try:
            cap.read()
            return f"prime_ok: {note}"
        finally:
            cap.release()

    def _try_open_camera(self) -> Tuple[Optional[cv2.VideoCapture], str]:
        """
        On macOS, CAP_AVFOUNDATION is typically the most reliable backend.
        We also auto-scan a few indices to handle non-zero default cameras.
        """

        indices = [self._camera_index]
        for idx in range(0, 4):
            if idx not in indices:
                indices.append(idx)

        backends = [None]
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            backends = [cv2.CAP_AVFOUNDATION, None]
        elif hasattr(cv2, "CAP_ANY"):
            backends = [cv2.CAP_ANY, None]

        last_note = "unknown"
        for idx in indices:
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_size[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_size[1])
                    if cap.isOpened():
                        note = f"camera_index={idx}, backend={backend if backend is not None else 'default'}"
                        return cap, note
                    cap.release()
                    last_note = f"failed camera_index={idx}, backend={backend if backend is not None else 'default'}"
                except Exception as e:
                    last_note = f"exception camera_index={idx}, backend={backend if backend is not None else 'default'}: {e}"
        return None, last_note

    def _run(self) -> None:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_draw_styles = mp.solutions.drawing_styles

        cap, note = self._try_open_camera()

        prev_time = time.time()

        with self._lock:
            self._running = True

        if not cap:
            with self._lock:
                self._counter.feedback = (
                    "Camera access denied or camera unavailable. "
                    "macOS: System Settings → Privacy & Security → Camera → enable Cursor (or Terminal). "
                    "Also close other apps using the camera (Zoom/Meet/FaceTime). "
                    f"({note})"
                )
                self._latest_snapshot = self._counter.snapshot(fps=0.0)
                self._running = False
            return

        # Reduce latency if backend supports it.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        try:
            with mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=self._min_det,
                min_tracking_confidence=self._min_track,
                model_complexity=self._model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
            ) as pose:
                frame_idx = 0
                while not self._stop.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        with self._lock:
                            self._latest_snapshot = self._counter.snapshot(fps=0.0)
                        time.sleep(0.05)
                        continue

                    frame = cv2.flip(frame, 1)

                    now = time.time()
                    fps = 1.0 / (now - prev_time + 1e-9)
                    prev_time = now

                    frame_idx += 1
                    results = None
                    if frame_idx % self._process_every_n == 0:
                        proc = frame
                        if (
                            self._process_size
                            and (proc.shape[1], proc.shape[0]) != self._process_size
                        ):
                            proc = cv2.resize(
                                proc,
                                self._process_size,
                                interpolation=cv2.INTER_LINEAR,
                            )

                        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
                        rgb.flags.writeable = False
                        results = pose.process(rgb)

                    if results and results.pose_landmarks:
                        if self._draw_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style(),
                            )
                        with self._lock:
                            self._counter.update(results.pose_landmarks.landmark)

                    with self._lock:
                        snap = self._counter.snapshot(fps=fps)
                        self._latest_snapshot = snap

                    out = frame
                    if self._stream_size and (
                        (out.shape[1], out.shape[0]) != self._stream_size
                    ):
                        out = cv2.resize(
                            out, self._stream_size, interpolation=cv2.INTER_AREA
                        )

                    ok, buf = cv2.imencode(
                        ".jpg",
                        out,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
                    )
                    if ok:
                        with self._lock:
                            self._latest_jpeg = buf.tobytes()
        finally:
            cap.release()
            with self._lock:
                self._running = False

