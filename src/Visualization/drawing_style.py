#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class ColorStyles:
    _RED = tuple(np.array([255, 48, 48]) / 255)
    _GREEN = tuple(np.array([48, 255, 48]) / 255)
    _BLUE = tuple(np.array([21, 101, 192]) / 255)
    _ORANGE = tuple(np.array([255, 127, 14]) / 255)
    _YELLOW = tuple(np.array([255, 204, 0]) / 255)

    CMAP_RGR = LinearSegmentedColormap.from_list("rgr", [_RED, _YELLOW, _GREEN, _GREEN, _YELLOW, _RED])
    CMAP_RG = LinearSegmentedColormap.from_list("rgg", [_RED, _ORANGE, _YELLOW, _GREEN, _GREEN])
    GRADIENT_STEP_SIZE = 100

    PLT_LINE_COLOR = (1, 1, 1)
    PLT_POINT_COLOR = _ORANGE
    PLT_TITLE_MAIN_COLOR = (0, 0, 0)


class DrawingStyles:
    THICKNESS_POSE_LANDMARKS = 100
    THICKNESS_LINES = 2
    THICKNESS_POINT_BORDER = 1
    Z_ORDER_LINES = 1
    Z_ORDER_POINTS = 2
    PLT_FONT_SIZE = 13
