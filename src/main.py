#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from constants import LeftBodyLandmarks
from src import body_position_analysis
import recommended_body_angles

video_path = "../example/example.mp4"

bike = recommended_body_angles.RoadBike(recommended_body_angles.RidingStyle.RACING)
body_position_analysis.bikefit_result(video_path, bike)
