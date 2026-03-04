import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = float(np.abs(np.degrees(radians)))
    return 360.0 - angle if angle > 180.0 else angle


def get_coords(landmarks, lm_enum) -> List[float]:
    lm = landmarks[lm_enum.value]
    return [float(lm.x), float(lm.y)]


def smooth_angle(angle_buffer: Deque[float], new_angle: float, window: int = 5) -> float:
    angle_buffer.append(float(new_angle))
    return float(np.mean(angle_buffer))


PL = mp.solutions.pose.PoseLandmark

EXERCISES: Dict[str, Dict[str, Any]] = {
    "Bicep Curl (Left)": {
        "joints": (PL.LEFT_SHOULDER, PL.LEFT_ELBOW, PL.LEFT_WRIST),
        "down_angle": 155,
        "up_angle": 45,
        "direction": "down_then_up",
    },
    "Bicep Curl (Right)": {
        "joints": (PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, PL.RIGHT_WRIST),
        "down_angle": 155,
        "up_angle": 45,
        "direction": "down_then_up",
    },
    "Push-Up": {
        "joints": (PL.LEFT_SHOULDER, PL.LEFT_ELBOW, PL.LEFT_WRIST),
        "down_angle": 75,
        "up_angle": 155,
        "direction": "up_then_down",
    },
    "Squat": {
        "joints": (PL.LEFT_HIP, PL.LEFT_KNEE, PL.LEFT_ANKLE),
        "down_angle": 95,
        "up_angle": 160,
        "direction": "up_then_down",
    },
    "Shoulder Press (Left)": {
        "joints": (PL.LEFT_ELBOW, PL.LEFT_SHOULDER, PL.LEFT_HIP),
        "down_angle": 70,
        "up_angle": 155,
        "direction": "down_then_up",
    },
    "Lateral Raise (Left)": {
        "joints": (PL.LEFT_HIP, PL.LEFT_SHOULDER, PL.LEFT_ELBOW),
        "down_angle": 30,
        "up_angle": 80,
        "direction": "down_then_up",
    },
}

HYSTERESIS = 10
HOLD_FRAMES = 3
COOLDOWN_SEC = 0.4


@dataclass
class RepSnapshot:
    exercise: str
    count: int
    stage: Optional[str]
    angle: float
    feedback: str
    started_at: float
    updated_at: float
    fps: float
    rep_durations: List[float]
    angle_mins: List[float]
    angle_maxs: List[float]

    def elapsed_sec(self) -> float:
        return max(0.0, self.updated_at - self.started_at)

    def reps_per_min(self) -> float:
        e = self.elapsed_sec()
        return (self.count / e) * 60.0 if e > 0.0 and self.count > 0 else 0.0

    def avg_cadence(self) -> float:
        return float(np.mean(self.rep_durations)) if self.rep_durations else 0.0

    def avg_rom(self) -> float:
        if not self.angle_mins or not self.angle_maxs:
            return 0.0
        diffs = [mx - mn for mx, mn in zip(self.angle_maxs, self.angle_mins)]
        return float(np.mean(diffs)) if diffs else 0.0


class RepCounter:
    def __init__(self, exercise_name: str):
        if exercise_name not in EXERCISES:
            raise ValueError(f"Unknown exercise: {exercise_name}")
        self.name = exercise_name
        self.config = EXERCISES[exercise_name]

        self.count = 0
        self.stage: Optional[str] = None  # 'start' | 'end'
        self.angle = 0.0

        self._angle_buf: Deque[float] = deque(maxlen=6)
        self._hold_counter = 0
        self._candidate: Optional[str] = None
        self._last_rep_time = 0.0

        self.rep_times: List[float] = []
        self.rep_durations: List[float] = []
        self.angle_mins: List[float] = []
        self.angle_maxs: List[float] = []
        self._rep_angle_min = 999.0
        self._rep_angle_max = 0.0
        self.started_at = time.time()

        self.feedback = "Get into position…"

    def update(self, landmarks) -> int:
        A, B, C = self.config["joints"]
        try:
            raw_angle = calculate_angle(
                get_coords(landmarks, A),
                get_coords(landmarks, B),
                get_coords(landmarks, C),
            )
        except Exception:
            return self.count

        self.angle = smooth_angle(self._angle_buf, raw_angle)
        self._rep_angle_min = min(self._rep_angle_min, self.angle)
        self._rep_angle_max = max(self._rep_angle_max, self.angle)

        self._detect_stage()
        return self.count

    def _detect_stage(self) -> None:
        direction = self.config["direction"]
        down_t = float(self.config["down_angle"])
        up_t = float(self.config["up_angle"])
        H = float(HYSTERESIS)

        new_candidate: Optional[str] = None
        if direction == "down_then_up":
            if self.angle > down_t + H:
                new_candidate = "start"
                self.feedback = "Lower ↓"
            elif self.angle < up_t - H:
                new_candidate = "end"
                self.feedback = "Hold… ✓"
        else:  # up_then_down
            if self.angle > up_t + H:
                new_candidate = "start"
                self.feedback = "Go down ↓"
            elif self.angle < down_t - H:
                new_candidate = "end"
                self.feedback = "Hold… ✓"

        if not new_candidate:
            self._hold_counter = 0
            return

        if new_candidate == self._candidate:
            self._hold_counter += 1
        else:
            self._candidate = new_candidate
            self._hold_counter = 1

        if self._hold_counter >= HOLD_FRAMES:
            self._commit_stage(new_candidate)

    def _commit_stage(self, new_stage: str) -> None:
        if new_stage == self.stage:
            return

        prev = self.stage
        self.stage = new_stage

        if new_stage == "end" and prev == "start":
            now = time.time()
            if now - self._last_rep_time < COOLDOWN_SEC:
                self.feedback = "Too fast — slow down!"
                self.stage = "start"
                return

            self.count += 1
            self._last_rep_time = now
            self.rep_times.append(now)
            self.feedback = f"Rep {self.count} ✓"

            if len(self.rep_times) >= 2:
                self.rep_durations.append(self.rep_times[-1] - self.rep_times[-2])

            self.angle_mins.append(self._rep_angle_min)
            self.angle_maxs.append(self._rep_angle_max)
            self._rep_angle_min = 999.0
            self._rep_angle_max = 0.0

    def reset(self) -> None:
        self.__init__(self.name)

    def snapshot(self, fps: float = 0.0) -> RepSnapshot:
        now = time.time()
        return RepSnapshot(
            exercise=self.name,
            count=int(self.count),
            stage=self.stage,
            angle=float(self.angle),
            feedback=str(self.feedback),
            started_at=float(self.started_at),
            updated_at=float(now),
            fps=float(fps),
            rep_durations=list(self.rep_durations),
            angle_mins=list(self.angle_mins),
            angle_maxs=list(self.angle_maxs),
        )

