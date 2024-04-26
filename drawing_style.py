#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Mapping

from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.pose import PoseLandmark
import matplotlib.colors


cmap_rgr = matplotlib.colors.LinearSegmentedColormap.from_list("",
            ["red", "yellow", "green", "green", "yellow","red"])
cmap_rg = matplotlib.colors.LinearSegmentedColormap.from_list("",
            ["red", "orange", "yellow", "green", "green"])

_RADIUS = 5
_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)
_CYAN = (192, 255, 48)
_MAGENTA = (192, 48, 255)

_THICKNESS_POSE_LANDMARKS = 8
_POSE_LANDMARKS = frozenset([
    PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE,
    PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR, PoseLandmark.MOUTH_LEFT,
    PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX, PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE,
    PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR,
    PoseLandmark.MOUTH_RIGHT, PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST,
    PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_FOOT_INDEX
])

def get_pose_landmarks_style() -> Mapping[int, DrawingSpec]:
    """Returns the default pose landmarks drawing style.
    
    Returns:
        A mapping from each pose landmark to its drawing spec.
    """
    pose_landmark_style = {}
    pose_spec = DrawingSpec(
        color=_BLUE, thickness=_THICKNESS_POSE_LANDMARKS)
    for landmark in _POSE_LANDMARKS:
      pose_landmark_style[landmark] = pose_spec
    pose_landmark_style[PoseLandmark.NOSE] = DrawingSpec(
        color=_BLUE, thickness=_THICKNESS_POSE_LANDMARKS)
    return pose_landmark_style
