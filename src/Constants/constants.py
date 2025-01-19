import enum
import os

from mediapipe.python.solutions.pose import PoseLandmark


class PathConstants:

    BASE_PATH = os.getcwd()
    ROOT_DIR_PATH = os.path.dirname(BASE_PATH)

    _MODEL_DIR_NAME = "models"
    _INPUT_DIR_NAME = "input"
    OUTPUT_DIR_NAME = "output"
    RUN_DIR_BASE_NAME = "run"
    FRAMES_DIR_NAME = ".frames"
    FRAME_FILE_NAME = "frame{:04d}.jpg"

    _INPUT_FILE_NAME = "example.mp4"
    _MODEL_FILE_NAME = "pose_landmarker_heavy.task"
    TOP_BOT_PEDAL_STROKE_FILE_NAME = "bikefit_top_vs_bottom_stroke.jpg"
    MEASURED_VS_RECOMMENDED_ANGLES_FILE_NAME = "measured_vs_recommended_angles.png"

    INPUT_DIR_PATH = os.path.join(ROOT_DIR_PATH, _INPUT_DIR_NAME)
    OUTPUT_DIR_PATH = os.path.join(ROOT_DIR_PATH, OUTPUT_DIR_NAME)
    MODEL_DIR_PATH = os.path.join(ROOT_DIR_PATH, _MODEL_DIR_NAME)

    MODEL_FILE_PATH = os.path.join(MODEL_DIR_PATH, _MODEL_FILE_NAME)
    INPUT_FILE_PATH = os.path.join(INPUT_DIR_PATH, _INPUT_FILE_NAME)


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
