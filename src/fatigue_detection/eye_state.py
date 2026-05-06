from .constants import (
    EAR_THRESHOLD,
    BLINK_MAX_FRAMES,
    DROWSY_MIN_FRAMES,
    EYE_STATUS_OPEN,
    EYE_STATUS_BLINKING,
    EYE_STATUS_CLOSED,
    EYE_STATUS_DROWSY,
)


class EyeClosureTracker:
    def __init__(self, ear_threshold=EAR_THRESHOLD, blink_max_frames=BLINK_MAX_FRAMES,
                 drowsy_min_frames=DROWSY_MIN_FRAMES):
        self.ear_threshold = ear_threshold
        self.blink_max_frames = blink_max_frames
        self.drowsy_min_frames = drowsy_min_frames
        self.closed_frames = 0
        self.status = EYE_STATUS_OPEN

    def update(self, avg_ear):
        if avg_ear < self.ear_threshold:
            self.closed_frames += 1
        else:
            self.closed_frames = 0

        if self.closed_frames == 0:
            self.status = EYE_STATUS_OPEN
        elif self.closed_frames <= self.blink_max_frames:
            self.status = EYE_STATUS_BLINKING
        elif self.closed_frames >= self.drowsy_min_frames:
            self.status = EYE_STATUS_DROWSY
        else:
            self.status = EYE_STATUS_CLOSED

        return self.status
