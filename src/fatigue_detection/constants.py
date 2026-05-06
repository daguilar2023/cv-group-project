EAR_THRESHOLD = 0.22
BLINK_MAX_FRAMES = 6
DROWSY_MIN_FRAMES = 15

EYE_STATUS_OPEN = "OPEN"
EYE_STATUS_BLINKING = "BLINKING"
EYE_STATUS_CLOSED = "CLOSED"
EYE_STATUS_DROWSY = "DROWSY_CLOSURE"

# MediaPipe Face Mesh eye landmark indices for EAR (6 points per eye)
# p1 = outer corner, p2 = upper-outer, p3 = upper-inner,
# p4 = inner corner, p5 = lower-inner, p6 = lower-outer
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# --- Mouth / MAR ---
MAR_NEUTRAL_OFFSET = 0.40
YAWN_MIN_FRAMES = 20

MOUTH_STATUS_NORMAL = "MOUTH_NORMAL"
MOUTH_STATUS_OPEN = "MOUTH_OPEN"
MOUTH_STATUS_YAWN = "YAWN_DETECTED"

# MediaPipe Face Mesh INNER lip landmark indices for MAR (8 points)
# p1 = left inner corner, p2-p4 = upper inner lip, p5 = right inner corner,
# p6-p8 = lower inner lip
MOUTH_INDICES = [78, 82, 13, 312, 308, 317, 14, 87]

# --- Head vertical droop (thresholds as fraction of face height) ---
VERTICAL_DROP_THRESHOLD = 0.10
HEAD_DEAD_ZONE = 0.03
HEAD_MOVEMENT_MIN_FRAMES = 8
SMOOTHING_ALPHA = 0.3

VERTICAL_STATUS_NORMAL = "VERTICAL_NORMAL"
VERTICAL_STATUS_DROOP = "HEAD_DROOP_DETECTED"

# --- Head yaw (turn left/right via facial geometry) ---
YAW_TURN_THRESHOLD = 0.15
YAW_DEAD_ZONE = 0.05
YAW_MIN_FRAMES = 6

YAW_STATUS_CENTERED = "CENTERED"
YAW_STATUS_TURN_LEFT = "TURN_LEFT"
YAW_STATUS_TURN_RIGHT = "TURN_RIGHT"

# Nose tip landmark index used as head reference point
NOSE_TIP_INDEX = 1

# Face height reference landmarks (forehead to chin)
FACE_TOP_INDEX = 10
FACE_BOTTOM_INDEX = 152

# Calibration
CALIBRATION_FRAMES = 30
