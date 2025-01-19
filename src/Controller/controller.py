import os
from math import inf

import mediapipe as mp

from Constants.bike import RoadBike, MountainBike
from Dataclasses.frame import Frame
from FrameEvaluation.frame_evaluator import FrameEvaluator
from VideoToolkit.video_tools import VideoTools
from Visualization.visualizer import Visualizer


class Controller:
    def __init__(self, bike: RoadBike | MountainBike):
        self.bike = bike

    def start(self):
        video_tools = VideoTools()
        video_tools.extract_images()
        frames_dir_path = video_tools.get_frames_folder_path()

        self._evaluate_and_annotate_frames(frames_dir_path)

    def _evaluate_and_annotate_frames(self, frames_dir_path):
        max_knee_angle, min_knee_angle = -inf, inf
        max_knee_angle_frame, min_knee_angle_frame = None, None
        torso_angles, shoulder_angles, elbow_angles = [], [], []
        for image_file_name in sorted(os.listdir(frames_dir_path)):
            image_path = os.path.join(frames_dir_path, image_file_name)
            # noinspection PyTypeChecker
            frame = Frame(
                name=image_file_name,
                image=mp.Image.create_from_file(image_path),
                path_to_image=image_path,
            )

            frame_evaluator = FrameEvaluator(frame=frame)
            body_position = frame_evaluator.get_body_positions_of_frame()
            body_angles = frame_evaluator.get_body_angles_of_frame(body_position=body_position)
            image_annotator = Visualizer(frame=frame, body_position=body_position, body_angles=body_angles)
            image_annotator.draw_landmarks_on_image_and_annotate()

            torso_angles.append(body_angles.torso_angle)
            shoulder_angles.append(body_angles.shoulder_angle)
            elbow_angles.append(body_angles.elbow_angle)

            if body_angles.knee_angle > max_knee_angle:
                max_knee_angle = body_angles.knee_angle
                max_knee_angle_frame = frame
            if body_angles.knee_angle < min_knee_angle:
                min_knee_angle = body_angles.knee_angle
                min_knee_angle_frame = frame

        Visualizer.save_image_of_top_and_bottom_pedal_stroke(
            top_frame=min_knee_angle_frame,
            bottom_frame=max_knee_angle_frame,
            output_dir_name=self._get_output_dir_path_from_frame(frame=frame),
        )
        avg_torso_angle = self._get_avg_of_list(torso_angles)
        avg_shoulder_angle = self._get_avg_of_list(shoulder_angles)
        avg_elbow_angle = self._get_avg_of_list(elbow_angles)
        Visualizer.save_image_of_angle_recommendations(
            bike=self.bike,
            min_knee_angle=min_knee_angle,
            max_knee_angle=max_knee_angle,
            avg_torso_angle=avg_torso_angle,
            avg_shoulder_angle=avg_shoulder_angle,
            avg_elbow_angle=avg_elbow_angle,
            output_dir_name=self._get_output_dir_path_from_frame(frame=frame),
        )

    @staticmethod
    def _get_avg_of_list(float_list: list[float]) -> float:
        return sum(float_list) / len(float_list)

    @staticmethod
    def _get_output_dir_path_from_frame(frame) -> os.PathLike | str:
        return os.path.dirname(os.path.dirname(frame.path_to_image))
