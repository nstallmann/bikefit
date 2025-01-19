import os
from dataclasses import dataclass

import mediapipe as mp


@dataclass
class Frame:
    name: str
    image: mp.Image
    path_to_image: os.PathLike | str
