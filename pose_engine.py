"""
Stateless MediaPipe frame processor.
The browser sends JPEG frames; this processes them and returns rep data + landmarks.
No camera capture on the server side — fully cloud-deployable.
"""

import threading
from typing import Dict, Optional, Any

import cv2
import mediapipe as mp
import numpy as np

from rep_counter_lib import EXERCISES, RepCounter


class FrameProcessor:
    """Thread-safe, stateless frame processor using browser-supplied JPEG frames."""

    def __init__(self, model_complexity: int = 0) -> None:
        self._lock = threading.Lock()
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self._current_exercise: str = list(EXERCISES.keys())[0]
        self._counters: Dict[str, RepCounter] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def set_exercise(self, name: str) -> None:
        if name in EXERCISES:
            with self._lock:
                self._current_exercise = name

    def reset(self) -> None:
        with self._lock:
            ex = self._current_exercise
            self._counters[ex] = RepCounter(ex)

    def process(self, jpeg_bytes: bytes, exercise: Optional[str] = None) -> Dict[str, Any]:
        """Decode a JPEG frame from the browser, run pose estimation, update rep counter."""
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Could not decode frame"}

        # Process in RGB (no server-side flip — browser mirrors via CSS)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        with self._lock:
            ex = exercise if exercise in EXERCISES else self._current_exercise
            counter = self._counters.setdefault(ex, RepCounter(ex))

            landmarks: list = []
            if results.pose_landmarks:
                counter.update(results.pose_landmarks.landmark)
                for lm in results.pose_landmarks.landmark:
                    landmarks.append({
                        "x": round(lm.x, 4),
                        "y": round(lm.y, 4),
                        "v": round(lm.visibility, 3),
                    })

            snap = counter.snapshot()
            return {
                "exercise": snap.exercise,
                "count": snap.count,
                "stage": snap.stage,
                "angle": round(snap.angle, 1),
                "feedback": snap.feedback,
                "elapsed_sec": round(snap.elapsed_sec(), 1),
                "rpm": round(snap.reps_per_min(), 1),
                "avg_cadence": round(snap.avg_cadence(), 2),
                "avg_rom": round(snap.avg_rom(), 1),
                "rep_durations": snap.rep_durations,
                "landmarks": landmarks,
            }

    def close(self) -> None:
        try:
            self._pose.close()
        except Exception:
            pass
