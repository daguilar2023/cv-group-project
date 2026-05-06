import argparse
import cv2
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from gesture_activation.detector import HandDetector
from gesture_activation.classifier import GestureClassifier
from gesture_activation.state_machine import GestureActivationStateMachine
from gesture_activation.constants import (
    STATE_WAITING_FOR_THUMBS_UP,
    STATE_WAITING_FOR_OPEN_PALM,
    STATE_ACTIVE,
    GESTURE_UNKNOWN,
    GESTURE_THUMBS_UP,
    GESTURE_OPEN_PALM,
)

from fatigue_detection.eye_detector import EyeDetector
from fatigue_detection.ear import compute_ear
from fatigue_detection.mar import compute_mar
from fatigue_detection.eye_state import EyeClosureTracker
from fatigue_detection.mouth_state import YawnTracker
from fatigue_detection.head_movement import HeadMovementTracker
from fatigue_detection.constants import (
    EYE_STATUS_DROWSY,
    MOUTH_STATUS_YAWN,
    VERTICAL_STATUS_DROOP,
    YAW_STATUS_CENTERED,
    YAW_STATUS_TURN_LEFT,
    YAW_STATUS_TURN_RIGHT,
)


def get_state_color(state):
    if state == STATE_WAITING_FOR_THUMBS_UP:
        return (100, 100, 255)
    elif state == STATE_WAITING_FOR_OPEN_PALM:
        return (0, 255, 255)
    elif state == STATE_ACTIVE:
        return (0, 255, 0)
    return (255, 255, 255)


def draw_ui(frame, state, gesture, message, fps, gesture_counter, hold_frames):
    color = get_state_color(state)

    cv2.putText(frame, f"State: {state}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Detected Gesture: {gesture}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"Instruction: {message}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Draw gesture confirmation progress bar
    if state != STATE_ACTIVE:
        progress = min(gesture_counter / hold_frames, 1.0)
        bar_x, bar_y, bar_w, bar_h = 20, 140, 300, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 2)
        fill_w = int(bar_w * progress)
        bar_color = (0, 255, 0) if progress >= 1.0 else (0, 200, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.putText(frame, f"{gesture_counter}/{hold_frames}", (bar_x + bar_w + 10, bar_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.putText(frame, "Press 'q' to quit | 'r' to reset",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return frame


def draw_fatigue_ui(frame, avg_ear, eye_status, closed_frames,
                    mar_value, mouth_status, mar_threshold,
                    norm_dy, vertical_status,
                    yaw_proxy, yaw_status,
                    calibrating):
    y = 220
    line = 28

    # --- Calibration banner ---
    if calibrating:
        cv2.putText(frame, "CALIBRATING... keep head still, mouth closed",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        return frame

    # --- Eye info ---
    eye_color = (0, 0, 255) if eye_status == EYE_STATUS_DROWSY else \
                (0, 165, 255) if eye_status == "CLOSED" else (0, 255, 0)

    cv2.putText(frame, f"EAR: {avg_ear:.2f}  Eye: {eye_status}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)

    # --- Mouth info (show raw MAR vs threshold) ---
    y += line
    thr_str = f"{mar_threshold:.2f}" if mar_threshold is not None else "--"
    mouth_color = (0, 0, 255) if mouth_status == MOUTH_STATUS_YAWN else (0, 255, 0)
    cv2.putText(frame, f"MAR: {mar_value:.2f} (thr {thr_str})  {mouth_status}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)

    # --- Head droop (vertical) ---
    y += line
    vert_color = (0, 0, 255) if vertical_status == VERTICAL_STATUS_DROOP else (0, 255, 0)
    cv2.putText(frame, f"Head dy: {norm_dy:+.3f}  {vertical_status}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vert_color, 2)

    # --- Head yaw (turn) ---
    y += line
    if yaw_status == YAW_STATUS_TURN_LEFT or yaw_status == YAW_STATUS_TURN_RIGHT:
        yaw_color = (0, 0, 255)
    else:
        yaw_color = (0, 255, 0)
    cv2.putText(frame, f"Yaw: {yaw_proxy:+.3f}  {yaw_status}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yaw_color, 2)

    # --- Alerts ---
    y += line + 5
    alerts = []
    if eye_status == EYE_STATUS_DROWSY:
        alerts.append("DROWSY EYES")
    if mouth_status == MOUTH_STATUS_YAWN:
        alerts.append("YAWN")
    if vertical_status == VERTICAL_STATUS_DROOP:
        alerts.append("HEAD DROOP")
    if yaw_status != YAW_STATUS_CENTERED:
        alerts.append(f"HEAD {yaw_status}")

    if alerts:
        alert_text = "!! " + " + ".join(alerts) + " !!"
        cv2.putText(frame, alert_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    return frame


def draw_hybrid_overlay(frame, cnn_class: str, cnn_conf: float,
                         gesture_method: str, source: str):
    """Small overlay showing what the CNN and hybrid logic decided."""
    h = frame.shape[0]
    y = h - 80
    cv2.putText(frame, f"CNN: {cnn_class} ({cnn_conf:.0%})  [{gesture_method}]",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    cv2.putText(frame, f"Hybrid source: {source}",
                (20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    return frame


def open_camera(preferred_indices=(0, 1, 2)):
    """
    Try common webcam indices and return the first available capture.
    """
    for idx in preferred_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Using webcam index {idx}")
            return cap
        cap.release()
    return None


def main():
    parser = argparse.ArgumentParser(description="Driver Fatigue Detection")
    parser.add_argument("--mode", choices=["classical", "hybrid"], default="classical",
                        help="classical = heuristic only (original); hybrid = CNN + heuristic")
    args = parser.parse_args()

    use_hybrid = args.mode == "hybrid"

    cap = open_camera()
    if cap is None:
        print("Error: Could not open webcam.")
        print("Tip: Allow camera access in macOS System Settings -> Privacy & Security -> Camera.")
        return

    detector      = HandDetector()
    heuristic_clf = GestureClassifier()
    state_machine = GestureActivationStateMachine()

    eye_detector = EyeDetector()
    eye_tracker  = EyeClosureTracker()
    yawn_tracker = YawnTracker()
    head_tracker = HeadMovementTracker()

    # ── Hybrid components (loaded lazily only in hybrid mode) ──────────────────
    hybrid_gesture  = None
    hybrid_fatigue  = None
    fatigue_smoother = None
    if use_hybrid:
        try:
            from ml.hybrid_detector import (HybridGestureClassifier,
                                             HybridFatigueClassifier,
                                             TemporalSmoother)
            from ml.dataset import FATIGUE_CLASSES
            hybrid_gesture   = HybridGestureClassifier()
            hybrid_fatigue   = HybridFatigueClassifier()
            fatigue_smoother = TemporalSmoother(len(FATIGUE_CLASSES), FATIGUE_CLASSES, window=15)
            cnn_g = "✓" if hybrid_gesture.cnn_available else "✗ (run train.py)"
            svm_g = "✓" if hybrid_gesture.svm_available else "✗ (run train_classical.py)"
            fat   = "✓" if hybrid_fatigue.available      else "✗ (run train.py)"
            print(f"\nHybrid mode — model status:")
            print(f"  Gesture CNN : {cnn_g}")
            print(f"  Gesture SVM : {svm_g}")
            print(f"  Fatigue CNN : {fat}")
        except ImportError as e:
            print(f"Warning: hybrid mode requires torch/sklearn ({e}). Falling back to classical.")
            use_hybrid = False

    prev_time = time.time()
    avg_ear   = 0.0
    mar_value = 0.0

    mode_label = "HYBRID" if use_hybrid else "CLASSICAL"
    print(f"\nDriver Fatigue Detection [{mode_label} mode]")
    print("Show THUMBS UP then OPEN PALM to activate.")
    print("Press 'q' to quit | 'r' to reset\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time

        gesture         = GESTURE_UNKNOWN
        gesture_method  = "heuristic"
        cnn_class       = "alert"
        cnn_conf        = 0.0
        hybrid_source   = "heuristic"

        if state_machine.state != STATE_ACTIVE:
            # ── Gesture activation phase ───────────────────────────────────────
            results   = detector.detect(frame)
            landmarks = detector.get_landmarks(results)
            frame     = detector.draw_landmarks(frame, results)

            if use_hybrid and hybrid_gesture is not None and (
                    hybrid_gesture.cnn_available or hybrid_gesture.svm_available):
                gesture_name, conf, gesture_method = hybrid_gesture.classify_hybrid(frame)
                # Map CNN class name → gesture constant
                if gesture_name == "thumbs_up":
                    gesture = GESTURE_THUMBS_UP
                elif gesture_name == "open_palm":
                    gesture = GESTURE_OPEN_PALM
                else:
                    gesture = GESTURE_UNKNOWN
            else:
                gesture        = heuristic_clf.classify(landmarks)
                gesture_method = "heuristic"

            state_machine.update(gesture)

        else:
            # ── Fatigue monitoring phase ───────────────────────────────────────
            face_result = eye_detector.detect(frame)
            frame       = eye_detector.draw_landmarks(frame, face_result)

            # Heuristic trackers (always run)
            left_eye, right_eye = eye_detector.get_eye_points(face_result)
            if left_eye is not None and right_eye is not None:
                avg_ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
                eye_tracker.update(avg_ear)

            mouth_pts = eye_detector.get_mouth_points(face_result)
            if mouth_pts is not None:
                mar_value = compute_mar(mouth_pts)
                yawn_tracker.update(mar_value)

            face_height = eye_detector.get_face_height(face_result, frame.shape)
            nose_x, nose_y = eye_detector.get_nose_tip(face_result, frame.shape)
            if nose_y is not None and face_height is not None:
                head_tracker.update(nose_y, face_height)

            yaw_pts = eye_detector.get_yaw_landmarks(face_result)
            if yaw_pts is not None:
                head_tracker.update_yaw(*yaw_pts)

            calibrating = not yawn_tracker.calibrated or not head_tracker.calibrated

            # Collect heuristic alerts
            heuristic_alerts = []
            if eye_tracker.status == EYE_STATUS_DROWSY:
                heuristic_alerts.append("DROWSY EYES")
            if yawn_tracker.status == MOUTH_STATUS_YAWN:
                heuristic_alerts.append("YAWN")
            if head_tracker.vertical_status == VERTICAL_STATUS_DROOP:
                heuristic_alerts.append("HEAD DROOP")
            if head_tracker.yaw_status != YAW_STATUS_CENTERED:
                heuristic_alerts.append(f"HEAD {head_tracker.yaw_status}")

            # CNN fatigue prediction (hybrid mode only)
            if use_hybrid and hybrid_fatigue is not None and hybrid_fatigue.available:
                # Use face crop instead of full frame for better CNN accuracy
                face_crop = eye_detector.get_face_crop(face_result, frame)
                cnn_input = face_crop if face_crop is not None else frame

                raw_proba = hybrid_fatigue._cnn.predict_proba(cnn_input)
                # Smooth over last 10 frames to suppress single-frame noise
                cnn_class, cnn_conf = fatigue_smoother.update(raw_proba)

                from ml.hybrid_detector import hybrid_fatigue_decision
                final_alerts, hybrid_source = hybrid_fatigue_decision(
                    heuristic_alerts, cnn_class, cnn_conf
                )
            else:
                final_alerts  = heuristic_alerts
                hybrid_source = "heuristic"

            frame = draw_fatigue_ui(
                frame, avg_ear, eye_tracker.status, eye_tracker.closed_frames,
                mar_value, yawn_tracker.status, yawn_tracker.mar_threshold,
                head_tracker.norm_dy, head_tracker.vertical_status,
                head_tracker.yaw_proxy, head_tracker.yaw_status,
                calibrating,
            )

            # Override the alert banner with merged hybrid alerts
            if not calibrating and final_alerts:
                alert_text = "!! " + " + ".join(final_alerts) + " !!"
                cv2.putText(frame, alert_text,
                            (20, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if use_hybrid:
                frame = draw_hybrid_overlay(
                    frame, cnn_class, cnn_conf, gesture_method, hybrid_source)

        # Mode label top-right
        cv2.putText(frame, f"[{mode_label}]",
                    (frame.shape[1] - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        frame = draw_ui(
            frame, state_machine.state, gesture,
            state_machine.message, fps,
            state_machine.gesture_counter, state_machine.hold_frames,
        )

        cv2.imshow("Driver Fatigue Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            state_machine.reset()
            print("System reset.")

    detector.release()
    eye_detector.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
