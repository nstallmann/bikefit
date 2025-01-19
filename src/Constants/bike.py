#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from enum import Enum


class RidingStyle(Enum):
    CASUAL = 0
    FITNESS = 1
    RACING = 2


class RoadBike:
    def __init__(self, riding_style: RidingStyle):
        self.riding_style = riding_style

        if riding_style == RidingStyle.CASUAL:
            # Pruitt & Matheny, 2006
            self.TORSO_HORIZONTAL_ANGLE_MIN = 50.0
            self.TORSO_HORIZONTAL_ANGLE_MAX = 60.0
        if riding_style == RidingStyle.FITNESS:
            # Pruitt & Matheny, 2006
            self.TORSO_HORIZONTAL_ANGLE_MIN = 40.0
            self.TORSO_HORIZONTAL_ANGLE_MAX = 50.0
        if riding_style == RidingStyle.RACING:
            # Pruitt & Matheny, 2006
            self.TORSO_HORIZONTAL_ANGLE_MIN = 30.0
            self.TORSO_HORIZONTAL_ANGLE_MAX = 45.0

    # Quesada et al, 2016
    SHOULDER_ANGLE_MIN = 75.0
    SHOULDER_ANGLE_MAX = 90.0

    # Millour et al, 2019
    MAXIMUM_KNEE_ANGLE_MIN = 137.0
    MAXIMUM_KNEE_ANGLE_MAX = 147.0

    # Scoz et al, 2021
    MINIMUM_KNEE_ANGLE_MIN = 68.0
    MINIMUM_KNEE_ANGLE_MAX = 74.0
    ANKLE_JOINT_MIN = 20.0
    ANKLE_JOINT_MAX = 30.0

    # Burt et al, 2014
    ELBOW_ANGLE_MIN = 150.0
    ELBOW_ANGLE_MAX = 160.0

    KNEE_OVER_TOE_VERTICAL_ANGLE = 0.0
    KNEE_OVER_TOE_VERTICAL_ANGLE_VARIANCE = 2.0


class MountainBike:
    # Scoz et al, 2021
    TORSO_HORIZONTAL_ANGLE_MIN = 50.0
    TORSO_HORIZONTAL_ANGLE_MAX = 65.0
    SHOULDER_ANGLE_MIN = 60.0
    SHOULDER_ANGLE_MAX = 70.0

    # Millour et al, 2019
    MAXIMUM_KNEE_ANGLE_MIN = 137.0
    MAXIMUM_KNEE_ANGLE_MAX = 147.0

    # Scoz et al, 2021
    MINIMUM_KNEE_ANGLE_MIN = 68.0
    MINIMUM_KNEE_ANGLE_MAX = 74.0
    ANKLE_JOINT_MIN = 20.0
    ANKLE_JOINT_MAX = 30.0

    # Burt et al, 2014
    ELBOW_ANGLE_MIN = 150.0
    ELBOW_ANGLE_MAX = 160.0

    KNEE_OVER_TOE_VERTICAL_ANGLE = 0.0
    KNEE_OVER_TOE_VERTICAL_ANGLE_VARIANCE = 2.0
