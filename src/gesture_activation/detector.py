import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections


class HandDetector:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "hand_landmarker.task"
        )
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._t0_ms = time.perf_counter() * 1000

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.perf_counter() * 1000 - self._t0_ms)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return result

    def draw_landmarks(self, frame, result):
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                )
        return frame

    def get_landmarks(self, result):
        if result.hand_landmarks:
            return result.hand_landmarks[0]
        return None

    def release(self):
        self.landmarker.close()
