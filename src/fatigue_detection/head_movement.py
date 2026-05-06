from .constants import (
    VERTICAL_DROP_THRESHOLD,
    HEAD_DEAD_ZONE,
    HEAD_MOVEMENT_MIN_FRAMES,
    SMOOTHING_ALPHA,
    CALIBRATION_FRAMES,
    VERTICAL_STATUS_NORMAL,
    VERTICAL_STATUS_DROOP,
    YAW_TURN_THRESHOLD,
    YAW_DEAD_ZONE,
    YAW_MIN_FRAMES,
    YAW_STATUS_CENTERED,
    YAW_STATUS_TURN_LEFT,
    YAW_STATUS_TURN_RIGHT,
)


class HeadMovementTracker:
    def __init__(self, vertical_drop_threshold=VERTICAL_DROP_THRESHOLD,
                 dead_zone=HEAD_DEAD_ZONE,
                 min_frames=HEAD_MOVEMENT_MIN_FRAMES,
                 alpha=SMOOTHING_ALPHA,
                 calibration_frames=CALIBRATION_FRAMES,
                 yaw_turn_threshold=YAW_TURN_THRESHOLD,
                 yaw_dead_zone=YAW_DEAD_ZONE,
                 yaw_min_frames=YAW_MIN_FRAMES):
        # --- Vertical droop ---
        self.vertical_drop_threshold = vertical_drop_threshold
        self.dead_zone = dead_zone
        self.min_frames = min_frames
        self.alpha = alpha
        self.calibration_frames = calibration_frames

        self.baseline_y = None
        self.smooth_y = None
        self.vertical_frames = 0
        self.vertical_status = VERTICAL_STATUS_NORMAL
        self.norm_dy = 0.0

        # Vertical calibration
        self._cal_samples = []
        self.calibrated = False

        # --- Yaw (head turn) ---
        self.yaw_turn_threshold = yaw_turn_threshold
        self.yaw_dead_zone = yaw_dead_zone
        self.yaw_min_frames = yaw_min_frames

        self.smooth_yaw = None
        self.yaw_proxy = 0.0
        self.yaw_frames = 0
        self.yaw_status = YAW_STATUS_CENTERED
        self._yaw_dir = YAW_STATUS_CENTERED

    # ---- Vertical droop (nose-y position in frame) ----

    def update(self, nose_y, face_height=1.0):
        # EMA smoothing
        if self.smooth_y is None:
            self.smooth_y = float(nose_y)
        else:
            self.smooth_y = self.alpha * nose_y + (1 - self.alpha) * self.smooth_y

        # Calibration phase: collect baseline samples
        if not self.calibrated:
            self._cal_samples.append(self.smooth_y)
            if len(self._cal_samples) >= self.calibration_frames:
                self.baseline_y = sum(self._cal_samples) / len(self._cal_samples)
                self.calibrated = True
            return 0.0, VERTICAL_STATUS_NORMAL

        # Normalize displacement by face height
        safe_height = max(face_height, 1.0)
        self.norm_dy = (self.smooth_y - self.baseline_y) / safe_height

        # Dead zone
        eff_dy = self.norm_dy if abs(self.norm_dy) > self.dead_zone else 0.0

        # Vertical movement (positive dy = head dropped down)
        if eff_dy > self.vertical_drop_threshold:
            self.vertical_frames += 1
        else:
            self.vertical_frames = 0

        if self.vertical_frames >= self.min_frames:
            self.vertical_status = VERTICAL_STATUS_DROOP
        else:
            self.vertical_status = VERTICAL_STATUS_NORMAL

        return self.norm_dy, self.vertical_status

    # ---- Yaw (head turn via facial geometry) ----

    def update_yaw(self, left_eye_cx, right_eye_cx, nose_x):
        eye_mid_x = (left_eye_cx + right_eye_cx) / 2.0
        inter_eye = abs(right_eye_cx - left_eye_cx)

        if inter_eye < 1e-6:
            return self.yaw_proxy, self.yaw_status

        raw_yaw = (nose_x - eye_mid_x) / inter_eye

        # EMA smoothing
        if self.smooth_yaw is None:
            self.smooth_yaw = raw_yaw
        else:
            self.smooth_yaw = self.alpha * raw_yaw + (1 - self.alpha) * self.smooth_yaw

        self.yaw_proxy = self.smooth_yaw

        # Dead zone
        eff_yaw = self.yaw_proxy if abs(self.yaw_proxy) > self.yaw_dead_zone else 0.0

        # Classification with sustained-frame requirement
        if eff_yaw > self.yaw_turn_threshold:
            self.yaw_frames += 1
            self._yaw_dir = YAW_STATUS_TURN_RIGHT
        elif eff_yaw < -self.yaw_turn_threshold:
            self.yaw_frames += 1
            self._yaw_dir = YAW_STATUS_TURN_LEFT
        else:
            self.yaw_frames = 0

        if self.yaw_frames >= self.yaw_min_frames:
            self.yaw_status = self._yaw_dir
        else:
            self.yaw_status = YAW_STATUS_CENTERED

        return self.yaw_proxy, self.yaw_status
