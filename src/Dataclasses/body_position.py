from dataclasses import dataclass

import numpy as np


@dataclass
class BodyPosition:
    shoulder_position_px: np.ndarray
    elbow_position_px: np.ndarray
    wrist_position_px: np.ndarray
    hip_position_px: np.ndarray
    knee_position_px: np.ndarray
    ankle_position_px: np.ndarray
    heel_position_px: np.ndarray
    toe_position_px: np.ndarray
