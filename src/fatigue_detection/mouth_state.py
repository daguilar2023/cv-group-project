from .constants import (
    MAR_NEUTRAL_OFFSET,
    YAWN_MIN_FRAMES,
    CALIBRATION_FRAMES,
    MOUTH_STATUS_NORMAL,
    MOUTH_STATUS_OPEN,
    MOUTH_STATUS_YAWN,
)


class YawnTracker:
    def __init__(self, neutral_offset=MAR_NEUTRAL_OFFSET, yawn_min_frames=YAWN_MIN_FRAMES,
                 calibration_frames=CALIBRATION_FRAMES):
        self.neutral_offset = neutral_offset
        self.yawn_min_frames = yawn_min_frames
        self.calibration_frames = calibration_frames

        self.open_frames = 0
        self.status = MOUTH_STATUS_NORMAL

        # Calibration state
        self._cal_samples = []
        self.neutral_mar = None
        self.mar_threshold = None
        self.calibrated = False

    def update(self, mar_value):
        # Calibration phase: collect neutral MAR samples
        if not self.calibrated:
            self._cal_samples.append(mar_value)
            if len(self._cal_samples) >= self.calibration_frames:
                self.neutral_mar = sum(self._cal_samples) / len(self._cal_samples)
                self.mar_threshold = self.neutral_mar + self.neutral_offset
                self.calibrated = True
            self.status = MOUTH_STATUS_NORMAL
            return self.status

        if mar_value > self.mar_threshold:
            self.open_frames += 1
        else:
            self.open_frames = 0

        if self.open_frames == 0:
            self.status = MOUTH_STATUS_NORMAL
        elif self.open_frames < self.yawn_min_frames:
            self.status = MOUTH_STATUS_OPEN
        else:
            self.status = MOUTH_STATUS_YAWN

        return self.status
