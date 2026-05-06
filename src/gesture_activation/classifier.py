from .constants import GESTURE_THUMBS_UP, GESTURE_OPEN_PALM, GESTURE_UNKNOWN


class GestureClassifier:
    def __init__(self):
        pass

    def classify(self, hand_landmarks):
        if hand_landmarks is None:
            return GESTURE_UNKNOWN

        landmarks = hand_landmarks
        finger_states = self._get_finger_states(landmarks)

        thumb, index, middle, ring, pinky = finger_states

        # Open palm: all five fingers extended
        if thumb and index and middle and ring and pinky:
            return GESTURE_OPEN_PALM

        # Thumbs up: thumb extended, all other fingers folded
        if thumb and not index and not middle and not ring and not pinky:
            return GESTURE_THUMBS_UP

        return GESTURE_UNKNOWN

    def _get_finger_states(self, landmarks):
        thumb_extended = self._is_thumb_extended(landmarks)
        index_extended = self._is_finger_extended(landmarks, 8, 6)
        middle_extended = self._is_finger_extended(landmarks, 12, 10)
        ring_extended = self._is_finger_extended(landmarks, 16, 14)
        pinky_extended = self._is_finger_extended(landmarks, 20, 18)
        return thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended

    def _is_finger_extended(self, landmarks, tip_id, pip_id):
        return landmarks[tip_id].y < landmarks[pip_id].y

    def _is_thumb_extended(self, landmarks):
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]

        # Thumb clearly vertical (not sideways)
        thumb_up = (
            thumb_tip.y < thumb_ip.y < thumb_mcp.y
        )

        # Thumb must be significantly above wrist (not just slightly)
        thumb_above_wrist = (
            thumb_tip.y < wrist.y - 0.08
        )

        # Thumb must NOT be too close to palm (avoid lazy fist)
        thumb_far_from_palm = (
            abs(thumb_tip.x - wrist.x) > 0.05
        )

        return (
            thumb_up and
            thumb_above_wrist and
            thumb_far_from_palm
        )
