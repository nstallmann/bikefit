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
            self.torso_horizontal_angle_hoods_lower = 50.
            self.torso_horizontal_angle_hoods_upper = 60.
        if riding_style == RidingStyle.FITNESS:
            # Pruitt & Matheny, 2006    
            self.torso_horizontal_angle_hoods_lower = 40.
            self.torso_horizontal_angle_hoods_upper = 50.
        if riding_style == RidingStyle.RACING:
            # Pruitt & Matheny, 2006    
            self.torso_horizontal_angle_hoods_lower = 30.
            self.torso_horizontal_angle_hoods_upper = 45.

    # Millour et al, 2019
    maximum_knee_angle_min = 137.
    maximum_knee_angle_max = 147.

    # Scoz et al, 2021
    minimum_knee_angle_min = 68.
    minimum_knee_angle_max = 74.
    ankle_joint_range_minimum = 20.
    ankle_joint_range_maximum = 30.


    
    # Quesada et al, 2016
    shoulder_angle_lower = 75.
    shoulder_angle_upper = 90.
    
    # Burt et al, 2014
    elbow_angle_lower = 150.
    elbow_angle_upper = 160.
    
    knee_over_toe_vertical_angle = 0.
    knee_over_toe_vertical_angle_variance = 2.


class MountainBike:
    # Millour et al, 2019
    maximum_knee_angle_min = 137.
    maximum_knee_angle_max = 147.

    # Scoz et al, 2021
    minimum_knee_angle_min = 68.
    minimum_knee_angle_max = 74.
    ankle_joint_range_minimum = 20.
    ankle_joint_range_maximum = 30.
    torso_horizontal_angle_hoods_lower = 50.
    torso_horizontal_angle_hoods_upper = 65.
    shoulder_angle_lower = 60.
    shoulder_angle_upper = 70.

    # Burt et al, 2014
    elbow_angle_lower = 150.
    elbow_angle_upper = 160.

    knee_over_toe_vertical_angle = 0.
    knee_over_toe_vertical_angle_variance = 2.
