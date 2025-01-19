from dataclasses import dataclass


@dataclass
class BodyAngles:
    knee_angle: float
    hip_angle: float
    elbow_angle: float
    shoulder_angle: float
    torso_angle: float