import enum

from mediapipe.python.solutions.pose import PoseLandmark


class LeftBodyLandmarks(enum.IntEnum):
    ELBOW = PoseLandmark.LEFT_ELBOW
    SHOULDER = PoseLandmark.LEFT_SHOULDER
    WRIST = PoseLandmark.LEFT_WRIST
    ANKLE = PoseLandmark.LEFT_ANKLE
    HEEL = PoseLandmark.LEFT_HEEL
    KNEE = PoseLandmark.LEFT_KNEE
    FOOT = PoseLandmark.LEFT_FOOT_INDEX
    HIP = PoseLandmark.LEFT_HIP
    
class RightBodyLandmarks(enum.IntEnum):
    ELBOW = PoseLandmark.RIGHT_ELBOW
    SHOULDER = PoseLandmark.RIGHT_SHOULDER
    WRIST = PoseLandmark.RIGHT_WRIST
    ANKLE = PoseLandmark.RIGHT_ANKLE
    HEEL = PoseLandmark.RIGHT_HEEL
    KNEE = PoseLandmark.RIGHT_KNEE
    FOOT = PoseLandmark.RIGHT_FOOT_INDEX
    HIP = PoseLandmark.RIGHT_HIP
