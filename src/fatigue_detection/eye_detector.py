import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

from .constants import (
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES,
    NOSE_TIP_INDEX, FACE_TOP_INDEX, FACE_BOTTOM_INDEX,
)


class EyeDetector:
    """Wraps MediaPipe FaceLandmarker and extracts eyes, mouth, and head landmarks."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task"
        )
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._t0_ms = time.perf_counter() * 1000

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.perf_counter() * 1000 - self._t0_ms)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return result

    def get_eye_points(self, result):
        if not result.face_landmarks:
            return None, None

        landmarks = result.face_landmarks[0]

        left_eye = [(landmarks[i].x, landmarks[i].y) for i in LEFT_EYE_INDICES]
        right_eye = [(landmarks[i].x, landmarks[i].y) for i in RIGHT_EYE_INDICES]

        return left_eye, right_eye

    def get_mouth_points(self, result):
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        mouth = [(landmarks[i].x, landmarks[i].y) for i in MOUTH_INDICES]
        return mouth

    def get_nose_tip(self, result, frame_shape):
        if not result.face_landmarks:
            return None, None

        landmarks = result.face_landmarks[0]
        h, w = frame_shape[:2]
        nose = landmarks[NOSE_TIP_INDEX]
        return int(nose.x * w), int(nose.y * h)

    def get_yaw_landmarks(self, result):
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        left_eye_cx = sum(landmarks[i].x for i in LEFT_EYE_INDICES) / len(LEFT_EYE_INDICES)
        right_eye_cx = sum(landmarks[i].x for i in RIGHT_EYE_INDICES) / len(RIGHT_EYE_INDICES)
        nose_x = landmarks[NOSE_TIP_INDEX].x

        return left_eye_cx, right_eye_cx, nose_x

    def get_face_height(self, result, frame_shape):
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h = frame_shape[0]
        top_y = landmarks[FACE_TOP_INDEX].y * h
        bot_y = landmarks[FACE_BOTTOM_INDEX].y * h
        return abs(bot_y - top_y)

    def draw_landmarks(self, frame, result):
        if not result.face_landmarks:
            return frame

        h, w = frame.shape[:2]
        landmarks = result.face_landmarks[0]

        # Eyes
        for indices, color in [(LEFT_EYE_INDICES, (0, 255, 0)), (RIGHT_EYE_INDICES, (0, 255, 0))]:
            pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
            for pt in pts:
                cv2.circle(frame, pt, 2, color, -1)

        # Mouth
        mouth_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH_INDICES]
        for pt in mouth_pts:
            cv2.circle(frame, pt, 2, (255, 0, 255), -1)

        # Nose tip
        nose = landmarks[NOSE_TIP_INDEX]
        cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 3, (255, 255, 0), -1)

        return frame

    # Keep old method name as alias for backward compatibility
    def draw_eye_landmarks(self, frame, result):
        return self.draw_landmarks(frame, result)

    def get_face_crop(self, result, frame, padding: float = 0.25):
        """
        Return a cropped BGR image of just the face region.
        padding: fraction of face size added on each side (default 25%).
        Returns None if no face detected.
        """
        if not result.face_landmarks:
            return None

        h, w = frame.shape[:2]
        landmarks = result.face_landmarks[0]

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        fw = max(xs) - min(xs)
        fh = max(ys) - min(ys)

        x_min = max(0, int((min(xs) - padding * fw) * w))
        x_max = min(w, int((max(xs) + padding * fw) * w))
        y_min = max(0, int((min(ys) - padding * fh) * h))
        y_max = min(h, int((max(ys) + padding * fh) * h))

        if x_max <= x_min or y_max <= y_min:
            return None

        return frame[y_min:y_max, x_min:x_max]

    def release(self):
        self.landmarker.close()
