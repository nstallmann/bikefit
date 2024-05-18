#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import frame_analysis
import recommended_pose

video_path = "example.mp4"

bike = recommended_pose.RoadBike(recommended_pose.RidingStyle.RACING)

frame_analysis.bikefit_result(video_path, bike)
