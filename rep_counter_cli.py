import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return 360 - angle if angle > 180 else angle

def get_coords(landmarks, lm_enum):
    lm = landmarks[lm_enum.value]
    return [lm.x, lm.y]

def smooth_angle(angle_buffer, new_angle, window=5):
    """Moving average to remove jitter."""
    angle_buffer.append(new_angle)
    return np.mean(angle_buffer)

# ─────────────────────────────────────────────
#  EXERCISE CONFIG
# ─────────────────────────────────────────────

PL = mp.solutions.pose.PoseLandmark

EXERCISES = {
    "Bicep Curl (Left)": {
        "joints": (PL.LEFT_SHOULDER, PL.LEFT_ELBOW, PL.LEFT_WRIST),
        "down_angle": 155, "up_angle": 45,
        "direction": "down_then_up",
    },
    "Bicep Curl (Right)": {
        "joints": (PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, PL.RIGHT_WRIST),
        "down_angle": 155, "up_angle": 45,
        "direction": "down_then_up",
    },
    "Push-Up": {
        "joints": (PL.LEFT_SHOULDER, PL.LEFT_ELBOW, PL.LEFT_WRIST),
        "down_angle": 75, "up_angle": 155,
        "direction": "up_then_down",
    },
    "Squat": {
        "joints": (PL.LEFT_HIP, PL.LEFT_KNEE, PL.LEFT_ANKLE),
        "down_angle": 95, "up_angle": 160,
        "direction": "up_then_down",
    },
    "Shoulder Press (Left)": {
        "joints": (PL.LEFT_ELBOW, PL.LEFT_SHOULDER, PL.LEFT_HIP),
        "down_angle": 70, "up_angle": 155,
        "direction": "down_then_up",
    },
    "Lateral Raise (Left)": {
        "joints": (PL.LEFT_HIP, PL.LEFT_SHOULDER, PL.LEFT_ELBOW),
        "down_angle": 30, "up_angle": 80,
        "direction": "down_then_up",
    },
}

# ─────────────────────────────────────────────
#  ACCURATE REP COUNTER
# ─────────────────────────────────────────────

HYSTERESIS   = 10    # degrees past threshold required to confirm stage
HOLD_FRAMES  = 3     # frames joint must stay in position to confirm stage
COOLDOWN_SEC = 0.4   # seconds between valid reps

class RepCounter:
    def __init__(self, exercise_name: str):
        if exercise_name not in EXERCISES:
            raise ValueError(f"Unknown exercise: {exercise_name}")
        self.name    = exercise_name
        self.config  = EXERCISES[exercise_name]
        self.count   = 0
        self.stage   = None       # 'start' | 'end'
        self.angle   = 0.0

        # Accuracy improvements
        self._angle_buf      = deque(maxlen=6)   # smoothing window
        self._hold_counter   = 0                  # frames in candidate stage
        self._candidate      = None               # pending stage
        self._last_rep_time  = 0.0

        # Analytics
        self.rep_times       = []        # timestamps of each rep
        self.rep_durations   = []        # seconds per rep (cadence)
        self.angle_mins      = []        # peak angle each rep (range of motion)
        self.angle_maxs      = []
        self._rep_angle_min  = 999.0
        self._rep_angle_max  = 0.0
        self.start_time      = time.time()
        self.feedback        = "Get into position…"

    # ── core update ──────────────────────────
    def update(self, landmarks):
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

        # Track range of motion within rep
        self._rep_angle_min = min(self._rep_angle_min, self.angle)
        self._rep_angle_max = max(self._rep_angle_max, self.angle)

        self._detect_stage()
        return self.count

    def _detect_stage(self):
        direction = self.config["direction"]
        down_t    = self.config["down_angle"]
        up_t      = self.config["up_angle"]
        H         = HYSTERESIS

        if direction == "down_then_up":
            if self.angle > down_t + H:
                new_candidate = "start"
                self.feedback = "Lower ↓"
            elif self.angle < up_t - H:
                new_candidate = "end"
                self.feedback = "Hold… ✓"
            else:
                self._hold_counter = 0
                return
        else:  # up_then_down
            if self.angle > up_t + H:
                new_candidate = "start"
                self.feedback = "Go down ↓"
            elif self.angle < down_t - H:
                new_candidate = "end"
                self.feedback = "Hold… ✓"
            else:
                self._hold_counter = 0
                return

        # Hysteresis hold: candidate must persist HOLD_FRAMES frames
        if new_candidate == self._candidate:
            self._hold_counter += 1
        else:
            self._candidate    = new_candidate
            self._hold_counter = 1

        if self._hold_counter >= HOLD_FRAMES:
            self._commit_stage(new_candidate)

    def _commit_stage(self, new_stage):
        if new_stage == self.stage:
            return   # no change

        prev = self.stage
        self.stage = new_stage

        # A rep is completed when we reach "end" coming from "start"
        if new_stage == "end" and prev == "start":
            now = time.time()
            # Cooldown check
            if now - self._last_rep_time < COOLDOWN_SEC:
                self.feedback = "Too fast — slow down!"
                self.stage = "start"   # reject this rep
                return

            self.count += 1
            self._last_rep_time = now
            self.rep_times.append(now)
            self.feedback = f"Rep {self.count} ✓"

            # Cadence
            if len(self.rep_times) >= 2:
                self.rep_durations.append(
                    self.rep_times[-1] - self.rep_times[-2]
                )

            # Range of motion
            self.angle_mins.append(self._rep_angle_min)
            self.angle_maxs.append(self._rep_angle_max)
            self._rep_angle_min = 999.0
            self._rep_angle_max = 0.0

    def reset(self):
        self.__init__(self.name)

    # ── analytics helpers ────────────────────
    def elapsed(self):
        return time.time() - self.start_time

    def avg_cadence(self):
        return np.mean(self.rep_durations) if self.rep_durations else 0.0

    def avg_rom(self):
        if self.angle_mins and self.angle_maxs:
            return np.mean([mx - mn for mx, mn in zip(self.angle_mins, self.angle_maxs)])
        return 0.0

    def reps_per_min(self):
        elapsed = self.elapsed()
        return (self.count / elapsed) * 60 if elapsed > 0 and self.count > 0 else 0.0


# ─────────────────────────────────────────────
#  ANALYTICS SUMMARY SCREEN
# ─────────────────────────────────────────────

def show_summary(frame, counter: RepCounter):
    """Draw a full-screen analytics overlay."""
    h, w = frame.shape[:2]

    # Blur background
    blurred = cv2.GaussianBlur(frame, (31, 31), 0)
    overlay = blurred.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.75, blurred, 0.25, 0, frame)

    # Title
    cv2.putText(frame, "── WORKOUT SUMMARY ──", (w//2 - 200, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 220, 50), 2)

    elapsed = counter.elapsed()
    mins, secs = divmod(int(elapsed), 60)

    stats = [
        ("Exercise",         counter.name),
        ("Total Reps",       str(counter.count)),
        ("Duration",         f"{mins}m {secs}s"),
        ("Reps / Min",       f"{counter.reps_per_min():.1f}"),
        ("Avg Cadence",      f"{counter.avg_cadence():.1f} sec/rep"),
        ("Avg Range of Motion", f"{counter.avg_rom():.1f}°"),
        ("Best Rep ROM",     f"{max([mx-mn for mx,mn in zip(counter.angle_maxs, counter.angle_mins)], default=0):.1f}°"),
        ("Completed At",     datetime.now().strftime("%H:%M:%S")),
    ]

    y = 110
    for label, value in stats:
        cv2.putText(frame, f"{label}:", (60, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1)
        cv2.putText(frame, value, (320, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 180), 2)
        y += 42

    # Per-rep timeline bar
    if counter.rep_durations:
        y += 10
        cv2.putText(frame, "Rep Pace Timeline:", (60, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
        y += 20
        bar_w = (w - 120) // max(len(counter.rep_durations), 1)
        max_d = max(counter.rep_durations) + 0.1
        for i, d in enumerate(counter.rep_durations):
            bh = int((d / max_d) * 60)
            bx = 60 + i * bar_w
            by = y + 60 - bh
            color = (0, 200, 100) if d < counter.avg_cadence() * 1.2 else (0, 100, 255)
            cv2.rectangle(frame, (bx, by), (bx + bar_w - 4, y + 60), color, -1)
            cv2.putText(frame, str(i + 1), (bx + 2, y + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.putText(frame, "Press N=Next Exercise  R=Restart  Q=Quit",
                (w//2 - 210, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
    return frame


# ─────────────────────────────────────────────
#  LIVE UI
# ─────────────────────────────────────────────

def draw_live_ui(frame, counter: RepCounter, fps: float):
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 115), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, counter.name, (15, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 220, 50), 2)
    cv2.putText(frame, f"Reps: {counter.count}", (15, 85),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 120), 3)

    # Right stats
    cv2.putText(frame, f"Angle: {int(counter.angle)}", (w - 210, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"FPS:   {fps:.0f}", (w - 210, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cadence = f"{counter.avg_cadence():.1f}s/rep" if counter.rep_durations else "--"
    cv2.putText(frame, f"Pace:  {cadence}", (w - 210, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1)

    # Stage pill
    stage_color = (0, 190, 0) if counter.stage == "end" else (0, 130, 255)
    label = counter.stage.upper() if counter.stage else "WAITING"
    cv2.rectangle(frame, (w - 210, 100), (w - 10, 114), stage_color, -1)
    cv2.putText(frame, label, (w - 205, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # Angle arc progress bar
    pct = min(max(counter.angle / 180.0, 0), 1)
    bar_x, bar_y, bar_h_max = 10, 130, h - 200
    filled = int(pct * bar_h_max)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 18, bar_y + bar_h_max), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y + bar_h_max - filled),
                  (bar_x + 18, bar_y + bar_h_max), (0, 200, 255), -1)
    cv2.putText(frame, "ROM", (bar_x - 2, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Bottom feedback bar
    cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, counter.feedback, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "S=Summary  N=Next  R=Reset  Q=Quit",
                (15, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    exercise_names = list(EXERCISES.keys())
    ex_index = 0
    counter  = RepCounter(exercise_names[ex_index])

    mp_pose        = mp.solutions.pose
    mp_drawing     = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌  Webcam not found.")
        return

    show_summary_screen = False
    prev_time = time.time()

    with mp_pose.Pose(
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
        model_complexity=1,
    ) as pose:

        print("✅  Rep Counter v2 running.")
        print("   S=Summary | N=Next exercise | R=Reset | Q=Quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            now  = time.time()
            fps  = 1.0 / (now - prev_time + 1e-9)
            prev_time = now

            if show_summary_screen:
                frame = show_summary(frame, counter)
                cv2.imshow("Rep Counter v2", frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') or key == 27:
                    show_summary_screen = False
                elif key == ord('r'):
                    counter.reset()
                    show_summary_screen = False
                elif key == ord('n'):
                    ex_index = (ex_index + 1) % len(exercise_names)
                    counter  = RepCounter(exercise_names[ex_index])
                    show_summary_screen = False
                continue

            # ── pose processing ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style(),
                )
                counter.update(results.pose_landmarks.landmark)

            draw_live_ui(frame, counter, fps)
            cv2.imshow("Rep Counter v2", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.reset()
            elif key == ord('n'):
                ex_index = (ex_index + 1) % len(exercise_names)
                counter  = RepCounter(exercise_names[ex_index])
            elif key == ord('s'):
                show_summary_screen = True

    cap.release()
    cv2.destroyAllWindows()

    # Print summary to terminal too
    print("\n══════════════════════════════")
    print(f"  WORKOUT SUMMARY — {counter.name}")
    print(f"  Total Reps     : {counter.count}")
    print(f"  Duration       : {int(counter.elapsed()//60)}m {int(counter.elapsed()%60)}s")
    print(f"  Reps/Min       : {counter.reps_per_min():.1f}")
    print(f"  Avg Cadence    : {counter.avg_cadence():.2f} sec/rep")
    print(f"  Avg ROM        : {counter.avg_rom():.1f}°")
    print("══════════════════════════════\n")


if __name__ == "__main__":
    main()

