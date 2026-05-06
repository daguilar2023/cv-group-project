import time

from .constants import (
    GESTURE_HOLD_FRAMES,
    GESTURE_OPEN_PALM,
    GESTURE_THUMBS_UP,
    GESTURE_UNKNOWN,
    NO_HAND_GRACE_FRAMES,
    SEQUENCE_TIMEOUT,
    STATE_ACTIVE,
    STATE_WAITING_FOR_OPEN_PALM,
    STATE_WAITING_FOR_THUMBS_UP,
)


class GestureActivationStateMachine:
    def __init__(self, hold_frames=GESTURE_HOLD_FRAMES, timeout=SEQUENCE_TIMEOUT,
                 grace_frames=NO_HAND_GRACE_FRAMES):
        self.hold_frames = hold_frames
        self.timeout = timeout
        self.grace_frames = grace_frames
        self.state = STATE_WAITING_FOR_THUMBS_UP
        self.sequence_start_time = None
        self.current_gesture = None
        self.gesture_counter = 0
        self.no_hand_counter = 0
        self.message = "Show thumbs up"

    def reset(self):
        self.state = STATE_WAITING_FOR_THUMBS_UP
        self.sequence_start_time = None
        self.current_gesture = None
        self.gesture_counter = 0
        self.no_hand_counter = 0
        self.message = "Show thumbs up"

    def _target_gesture(self):
        if self.state == STATE_WAITING_FOR_THUMBS_UP:
            return GESTURE_THUMBS_UP
        elif self.state == STATE_WAITING_FOR_OPEN_PALM:
            return GESTURE_OPEN_PALM
        return None

    def update(self, detected_gesture):
        target = self._target_gesture()

        # Handle temporary hand loss / unknown with grace period
        if detected_gesture is None or detected_gesture == GESTURE_UNKNOWN:
            self.no_hand_counter += 1
            if self.no_hand_counter > self.grace_frames:
                self.gesture_counter = 0
            return "WAITING"

        self.no_hand_counter = 0

        # Increment counter when target gesture is detected, decrement otherwise
        if detected_gesture == target:
            self.gesture_counter += 1
        else:
            self.gesture_counter = max(0, self.gesture_counter - 1)

        if self.state == STATE_WAITING_FOR_THUMBS_UP:
            self.message = "Show thumbs up"
            if detected_gesture == GESTURE_THUMBS_UP and self.gesture_counter >= self.hold_frames:
                self.sequence_start_time = time.time()
                self.state = STATE_WAITING_FOR_OPEN_PALM
                self.gesture_counter = 0
                self.message = "Step 1 complete - now show open palm"
                return "STEP_1_COMPLETE"

        elif self.state == STATE_WAITING_FOR_OPEN_PALM:
            if self.sequence_start_time is not None:
                elapsed = time.time() - self.sequence_start_time
                if elapsed > self.timeout:
                    self.reset()
                    self.message = "Sequence timeout - try again"
                    return "TIMEOUT"

            if detected_gesture == GESTURE_OPEN_PALM and self.gesture_counter >= self.hold_frames:
                self.state = STATE_ACTIVE
                self.message = "Activation successful"
                return "ACTIVATED"

            self.message = "Step 1 complete - now show open palm"

        elif self.state == STATE_ACTIVE:
            self.message = "System active - fatigue monitoring"

        return "WAITING"
