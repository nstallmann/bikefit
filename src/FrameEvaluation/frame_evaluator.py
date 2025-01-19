#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from Dataclasses.body_angles import BodyAngles
from Dataclasses.body_position import BodyPosition
from Dataclasses.frame import Frame
from Constants.constants import LeftBodyLandmarks, PathConstants, RightBodyLandmarks


class FrameEvaluator:
    def __init__(self, frame: Frame):
        self.frame = frame
        self.left_side_front: Optional[bool] = None

    def get_body_positions_of_frame(self) -> BodyPosition:
        landmarks = self._get_landmarks_of_image()
        self.left_side_front = self._left_side_faces_camera(landmarks)
        if self.left_side_front:
            BodyLm = LeftBodyLandmarks
        else:
            BodyLm = RightBodyLandmarks

        img_res = self._get_image_resolution()

        return BodyPosition(
            shoulder_position_px=np.array((landmarks[BodyLm.SHOULDER].x, landmarks[BodyLm.SHOULDER].y)) * img_res,
            elbow_position_px=np.array((landmarks[BodyLm.ELBOW].x, landmarks[BodyLm.ELBOW].y)) * img_res,
            wrist_position_px=np.array((landmarks[BodyLm.WRIST].x, landmarks[BodyLm.WRIST].y)) * img_res,
            hip_position_px=np.array((landmarks[BodyLm.HIP].x, landmarks[BodyLm.HIP].y)) * img_res,
            knee_position_px=np.array((landmarks[BodyLm.KNEE].x, landmarks[BodyLm.KNEE].y)) * img_res,
            ankle_position_px=np.array((landmarks[BodyLm.ANKLE].x, landmarks[BodyLm.ANKLE].y)) * img_res,
            heel_position_px=np.array((landmarks[BodyLm.HEEL].x, landmarks[BodyLm.HEEL].y)) * img_res,
            toe_position_px=np.array((landmarks[BodyLm.FOOT].x, landmarks[BodyLm.FOOT].y)) * img_res,
        )

    def get_body_angles_of_frame(self, body_position: BodyPosition) -> BodyAngles:
        hip2knee = self._normalize(body_position.knee_position_px - body_position.hip_position_px)
        knee2ankle = self._normalize(body_position.ankle_position_px - body_position.knee_position_px)
        hip2shoulder = self._normalize(body_position.shoulder_position_px - body_position.hip_position_px)
        elbow2shoulder = self._normalize(body_position.shoulder_position_px - body_position.elbow_position_px)
        elbow2wrist = self._normalize(body_position.wrist_position_px - body_position.elbow_position_px)
        horizontal_line = np.array([1, 0])
        if self.left_side_front:
            horizontal_line *= -1

        return BodyAngles(
            knee_angle=180 - self._angle_between_vectors(hip2knee, knee2ankle),
            hip_angle=self._angle_between_vectors(hip2knee, hip2shoulder),
            elbow_angle=self._angle_between_vectors(elbow2shoulder, elbow2wrist),
            shoulder_angle=self._angle_between_vectors(hip2shoulder, elbow2shoulder),
            torso_angle=self._angle_between_vectors(hip2shoulder, horizontal_line),
        )

    # region Image Helper Functions
    def _get_image_resolution(self) -> np.ndarray:
        image_resolution = np.flip(self.frame.image.numpy_view().shape[:2])
        return image_resolution

    def _get_landmarks_of_image(self) -> list[NormalizedLandmark]:
        base_options = mp.tasks.BaseOptions
        pose_landmarker = mp.tasks.vision.PoseLandmarker
        pose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
        vision_running_mode = mp.tasks.vision.RunningMode
        options = pose_landmarker_options(
            base_options=base_options(model_asset_path=PathConstants.MODEL_FILE_PATH),
            running_mode=vision_running_mode.IMAGE,
        )
        with pose_landmarker.create_from_options(options) as landmarker:
            pose_landmarker_result = landmarker.detect(self.frame.image)
        return pose_landmarker_result.pose_landmarks[0]

    @staticmethod
    def _left_side_faces_camera(landmark_list: list[NormalizedLandmark]) -> bool:
        return landmark_list[PoseLandmark.LEFT_SHOULDER].z < landmark_list[PoseLandmark.RIGHT_SHOULDER].z

    # endregion Image Helper Functions

    # region Geometry Helper Functions
    @staticmethod
    def _normalize(np_array: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(np_array)
        if norm == 0:
            return np_array
        return np_array / norm

    @staticmethod
    def _rad_to_degree(angle_rad: float) -> float:
        return angle_rad / np.pi * 180

    def _angle_between_vectors(self, vector_1: np.ndarray, vector_2: np.ndarray) -> float:
        scalar = np.arccos(vector_1 @ vector_2)
        if scalar < 0:
            return 90 + self._rad_to_degree(scalar)
        else:
            return self._rad_to_degree(scalar)

    # endregion Geometry Helper Functions
