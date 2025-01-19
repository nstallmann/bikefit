from dataclasses import dataclass


@dataclass
class BodyAngle:
    frame_id: str
    def __init__(self, frame_id: str):
        self.frame_id = frame_id
    knee_angle: float
    hip_angle: float
    elbow_angle: float
    shoulder_angle: float