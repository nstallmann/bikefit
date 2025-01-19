#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path, PurePath
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from matplotlib import pyplot as plt
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from Dataclasses.body_angles import BodyAngles
from Dataclasses.body_position import BodyPosition
from Dataclasses.frame import Frame
from Visualization import drawing_style
from VideoToolkit import video_tools
from Constants.constants import LeftBodyLandmarks, PathConstants, RightBodyLandmarks


class ImageEvaluator:
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
        print(np.array(landmarks[BodyLm.SHOULDER].x, landmarks[BodyLm.SHOULDER].y) * img_res)

        return BodyPosition(
            shoulder_position_px=np.array(landmarks[BodyLm.SHOULDER].x, landmarks[BodyLm.SHOULDER].y) * img_res,
            elbow_position_px=np.array(landmarks[BodyLm.ELBOW].x, landmarks[BodyLm.ELBOW].y) * img_res,
            wrist_position_px=np.array(landmarks[BodyLm.WRIST].x, landmarks[BodyLm.WRIST].y) * img_res,
            hip_position_px=np.array(landmarks[BodyLm.HIP].x, landmarks[BodyLm.HIP].y) * img_res,
            knee_position_px=np.array(landmarks[BodyLm.KNEE].x, landmarks[BodyLm.KNEE].y) * img_res,
            ankle_position_px=np.array(landmarks[BodyLm.ANKLE].x, landmarks[BodyLm.ANKLE].y) * img_res,
            heel_position_px=np.array(landmarks[BodyLm.HEEL].x, landmarks[BodyLm.HEEL].y) * img_res,
            toe_position_px=np.array(landmarks[BodyLm.FOOT].x, landmarks[BodyLm.FOOT].y) * img_res,
        )

    def get_body_angles_of_frame(self, body_position: BodyPosition) -> BodyAngles:
        hip2knee = self._normalize(body_position.knee_position_px - body_position.hip_position_px)
        knee2ankle = self._normalize(body_position.ankle_position_px - body_position.knee_position_px)
        hip2shoulder = self._normalize(body_position.shoulder_position_px - body_position.hip_position_px)
        shoulder2elbow = self._normalize(body_position.elbow_position_px - body_position.shoulder_position_px)
        elbow2wrist = self._normalize(body_position.wrist_position_px - body_position.elbow_position_px)
        horizontal_line = np.array([1, 0])
        if not self.left_side_front:
            horizontal_line *= -1

        print(180 - self._angle_between_vectors(hip2knee, knee2ankle))
        return BodyAngles(
            knee_angle=180 - self._angle_between_vectors(hip2knee, knee2ankle),
            hip_angle=self._angle_between_vectors(hip2knee, hip2shoulder),
            elbow_angle=self._angle_between_vectors(shoulder2elbow, elbow2wrist),
            shoulder_angle=self._angle_between_vectors(hip2shoulder, shoulder2elbow),
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
        scalar = self._rad_to_degree(np.arccos(vector_1 @ vector_2))
        if scalar < 0:
            return 180 - self._rad_to_degree(scalar)
        else:
            return self._rad_to_degree(scalar)

    # endregion Geometry Helper Functions


class ImageAnnotator:
    def __init__(self):
        pass

    def draw_landmarks_on_image(self, rgb_image, detection_result, angle_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)
        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in pose_landmarks
                ]
            )
            pose_connections = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                pose_connections,
                drawing_style.get_pose_landmarks_style(),
            )

            # Legend with corresponding pose angles
            y_pixels = rgb_image.shape[0]
            font = 0
            fontScale = 0.8
            thickness = 2
            color = (0, 0, 0)
            spacing = y_pixels // 20
            annotated_image = cv2.putText(
                annotated_image,
                f"knee angle: {round(angle_result[0], 1)}",
                (10, 1 * spacing),
                fontFace=font,
                fontScale=fontScale,
                thickness=thickness,
                color=color,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"torso angle: {round(angle_result[1], 1)}",
                (10, 2 * spacing),
                fontFace=font,
                fontScale=fontScale,
                thickness=thickness,
                color=color,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"hip angle: {round(angle_result[2], 1)}",
                (10, 3 * spacing),
                fontFace=font,
                fontScale=fontScale,
                thickness=thickness,
                color=color,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"shoulder angle: {round(angle_result[3], 1)}",
                (10, 4 * spacing),
                fontFace=font,
                fontScale=fontScale,
                thickness=thickness,
                color=color,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"elbow angle: {round(angle_result[4], 1)}",
                (10, 5 * spacing),
                fontFace=font,
                fontScale=fontScale,
                thickness=thickness,
                color=color,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"KOPS angle: {round(angle_result[5], 1)}",
                (10, 6 * spacing),
                fontFace=font,
                fontScale=fontScale,
                thickness=thickness,
                color=color,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"foot angle: {round(angle_result[6], 1)}",
                (10, 7 * spacing),
                fontFace=font,
                fontScale=fontScale,
                thickness=thickness,
                color=color,
            )

            return annotated_image


def video_frame_angles(video_file):
    video_base_name = os.path.basename(video_file).split(".")[0]
    if os.path.isfile(os.path.join(video_base_name, "body_angles.npy")):
        return
    Path(video_base_name).mkdir(parents=True, exist_ok=True)
    frames_folder_path = os.path.join(video_base_name, "frames")
    video_tools.extract_images(video_file, frames_folder_path)

    angles = []
    for frame in sorted(os.listdir(frames_folder_path)):
        frame_path = os.path.join(frames_folder_path, frame)
        mp_image = mp.Image.create_from_file(frame_path)
        image_angles = body_pose_angles(mp_image)
        angles.append(image_angles)
        ### Annotate frame and save
        landmarks = grab_landmarks(mp_image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), landmarks, image_angles)
        rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(frame_path, rgb_image)

    angles = np.array(angles)
    np.save(f"{os.path.join(video_base_name, 'body_angles')}.npy", angles)


def bikefit_angles(video):
    video_frame_angles(video)
    video_base_name = Path(video).stem

    angles = np.load(PurePath(video_base_name, "body_angles.npy"))

    ## knee angles
    maximum_knee_angle = round(np.max(angles[:, 0]), 1)
    maximum_knee_angle_frame = np.argmax(angles[:, 0])

    minimum_knee_angle = round(np.min(angles[:, 0]), 1)
    minimum_knee_angle_frame = np.argmin(angles[:, 0])

    ## Torso angles
    avg_torso_angle = round(np.average(angles[:, 1]), 1)
    torso_angle_variance = round(np.var(angles[:, 1]), 1)

    ## Shoulder and arm angles
    avg_shoulder_angle = round(np.average(angles[:, 3]), 1)
    avg_arm_angle = round(np.average(angles[:, 4]), 1)
    arm_angle_variance = round(np.var(angles[:, 4]), 1)

    ### Foot recognition too imprecise for meaningful interpretations
    ankle_joint_min = round(np.min(angles[:, 6]), 1)
    ankle_joint_argmin = round(np.argmin(angles[:, 6]), 1)
    ankle_joint_max = round(np.max(angles[:, 6]), 1)
    ankle_joint_argmax = round(np.argmax(angles[:, 6]), 1)
    ankle_joint_range = round(ankle_joint_max - ankle_joint_min, 1)

    return [
        [
            maximum_knee_angle,
            minimum_knee_angle,
            avg_torso_angle,
            avg_shoulder_angle,
            avg_arm_angle,
        ],
        [maximum_knee_angle_frame, minimum_knee_angle_frame],
    ]


def show_pedal_strokes(left_frame, right_frame, save=True):
    rgb_image_top = cv2.imread(left_frame)
    rgb_image_bot = cv2.imread(right_frame)
    image = np.concatenate((rgb_image_top, rgb_image_bot), axis=1)
    if save:
        dir_name = Path(left_frame).parent.parent
        cv2.imwrite(os.path.join(dir_name, "bikefit_top_vs_bottom_stroke.jpg"), image)
        return

    cv2.imshow("Top vs Bottom pedal stroke", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_angles(bikefit_angles, bike, save_dir):
    """
    Args: bikefit_angles (list):
               list with entries
                    [max_knee_angle,
                     min_knee_angle,
                     torso_angle,
                     shoulder_angle,
                     elbow_angle]
          bike (Bike)
    """

    fig, axs = plt.subplots(nrows=5)
    fig.tight_layout()

    for ax in axs:
        _set_ax_params(ax)
    stepsize = 100

    ### Maximum Knee Angle
    axs[0].set_title("Maximum Knee Angle")
    left = bike.MAXIMUM_KNEE_ANGLE_MIN - 1
    right = bike.MAXIMUM_KNEE_ANGLE_MAX + 1
    gradient = np.linspace(left, right, stepsize)
    gradient = np.vstack((gradient, gradient))
    axs[0].imshow(gradient, aspect="auto", cmap=drawing_style.CMAP_RGR)
    max_knee_angle = max(bikefit_angles[0], left)
    max_knee_angle = min(max_knee_angle, right)
    line = (max_knee_angle - left) * (stepsize - 1) / (right - left)
    axs[0].vlines(line, ymin=0, ymax=1, color="black", linewidth=4)
    axs[0].set_ylim([0, 1])
    axs[0].text(
        -0.01,
        0.5,
        f"{left}°",
        horizontalalignment="right",
        verticalalignment="center",
        transform=axs[0].transAxes,
    )
    axs[0].text(
        1.01,
        0.5,
        f"{right}°",
        horizontalalignment="left",
        verticalalignment="center",
        transform=axs[0].transAxes,
    )
    axs[0].text(
        0.71,
        1.22,
        f"{bikefit_angles[0]}°",
        fontsize=12,
        color=drawing_style.CMAP_RGR(line / stepsize),
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=axs[0].transAxes,
    )

    ### Minimum Knee Angle
    axs[1].set_title("Minimum Knee Angle")
    left = bike.MINIMUM_KNEE_ANGLE_MIN - 1
    right = bike.MINIMUM_KNEE_ANGLE_MAX + 1
    gradient = np.linspace(left, right, stepsize)
    gradient = np.vstack((gradient, gradient))
    axs[1].imshow(gradient, aspect="auto", cmap=drawing_style.CMAP_RGR)
    min_knee_angle = max(bikefit_angles[1], left)
    min_knee_angle = min(min_knee_angle, right)
    line = (min_knee_angle - left) * (stepsize - 1) / (right - left)
    axs[1].vlines(line, ymin=0, ymax=1, color="black", linewidth=4)
    axs[1].set_ylim([0, 1])
    axs[1].text(
        -0.01,
        0.5,
        f"{left}°",
        horizontalalignment="right",
        verticalalignment="center",
        transform=axs[1].transAxes,
    )
    axs[1].text(
        1.01,
        0.5,
        f"{right}°",
        horizontalalignment="left",
        verticalalignment="center",
        transform=axs[1].transAxes,
    )
    axs[1].text(
        0.71,
        1.22,
        f"{bikefit_angles[1]}°",
        fontsize=12,
        color=drawing_style.CMAP_RGR(line / stepsize),
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=axs[1].transAxes,
    )

    ### Torso Angle
    axs[2].set_title("Average Torso Angle")
    left = bike.TORSO_HORIZONTAL_ANGLE_MIN - 1
    right = bike.TORSO_HORIZONTAL_ANGLE_MAX + 1
    gradient = np.linspace(left, right, stepsize)
    gradient = np.vstack((gradient, gradient))
    axs[2].imshow(gradient, aspect="auto", cmap=drawing_style.CMAP_RGR)
    torso_angle = max(bikefit_angles[2], left)
    torso_angle = min(torso_angle, right)
    line = (torso_angle - left) * (stepsize - 1) / (right - left)
    axs[2].vlines(line, ymin=0, ymax=1, color="black", linewidth=4)
    axs[2].set_ylim([0, 1])
    axs[2].text(
        -0.01,
        0.5,
        f"{left}°",
        horizontalalignment="right",
        verticalalignment="center",
        transform=axs[2].transAxes,
    )
    axs[2].text(
        1.01,
        0.5,
        f"{right}°",
        horizontalalignment="left",
        verticalalignment="center",
        transform=axs[2].transAxes,
    )
    axs[2].text(
        0.7,
        1.22,
        f"{bikefit_angles[2]}°",
        fontsize=12,
        color=drawing_style.CMAP_RGR(line / stepsize),
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=axs[2].transAxes,
    )

    ### Shoulder Angle
    axs[3].set_title("Average Shoulder Angle")
    left = bike.SHOULDER_ANGLE_MIN - 1
    right = bike.SHOULDER_ANGLE_MAX + 1
    gradient = np.linspace(left, right, stepsize)
    gradient = np.vstack((gradient, gradient))
    axs[3].imshow(gradient, aspect="auto", cmap=drawing_style.CMAP_RGR)
    shoulder_angle = max(bikefit_angles[3], left)
    shoulder_angle = min(shoulder_angle, right)
    line = (shoulder_angle - left) * (stepsize - 1) / (right - left)
    axs[3].vlines(line, ymin=0, ymax=1, color="black", linewidth=4)
    axs[3].set_ylim([0, 1])
    axs[3].text(
        -0.01,
        0.5,
        f"{left}°",
        horizontalalignment="right",
        verticalalignment="center",
        transform=axs[3].transAxes,
    )
    axs[3].text(
        1.01,
        0.5,
        f"{right}°",
        horizontalalignment="left",
        verticalalignment="center",
        transform=axs[3].transAxes,
    )
    axs[3].text(
        0.73,
        1.22,
        f"{bikefit_angles[3]}°",
        fontsize=12,
        color=drawing_style.CMAP_RGR(line / stepsize),
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=axs[3].transAxes,
    )

    ### Elbow Angle
    axs[4].set_title("Average Elbow Angle")
    left = bike.ELBOW_ANGLE_MIN - 1
    right = bike.ELBOW_ANGLE_MAX + 1
    gradient = np.linspace(left, right, stepsize)
    gradient = np.vstack((gradient, gradient))
    axs[4].imshow(gradient, aspect="auto", cmap=drawing_style.CMAP_RGR)
    elbow_angle = max(bikefit_angles[4], left)
    elbow_angle = min(elbow_angle, right)
    line = (elbow_angle - left) * (stepsize - 1) / (right - left)
    axs[4].vlines(line, ymin=0, ymax=1, color="black", linewidth=4)
    axs[4].set_ylim([0, 1])
    axs[4].text(
        -0.01,
        0.5,
        f"{left}°",
        horizontalalignment="right",
        verticalalignment="center",
        transform=axs[4].transAxes,
    )
    axs[4].text(
        1.01,
        0.5,
        f"{right}°",
        horizontalalignment="left",
        verticalalignment="center",
        transform=axs[4].transAxes,
    )
    axs[4].text(
        0.71,
        1.22,
        f"{bikefit_angles[4]}°",
        fontsize=12,
        color=drawing_style.CMAP_RGR(line / stepsize),
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=axs[4].transAxes,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "measured_vs_recommended_angles.png"), dpi=300)


def _set_ax_params(ax):
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )


def bikefit_result(video, bike):
    angles, frames = bikefit_angles(video)
    video_base_name = Path(video).stem
    visualize_angles(angles, bike, video_base_name)

    frame_path = os.path.join(video_base_name, "frames")
    left_frame = os.path.join(frame_path, f"frame{frames[1]:04d}.jpg")
    right_frame = os.path.join(frame_path, f"frame{frames[0]:04d}.jpg")
    show_pedal_strokes(left_frame, right_frame)

    print("\n\n\n")
    print(f"Your maximum knee angle is {angles[0]}.")
    print(f"Recommended range is {bike.MAXIMUM_KNEE_ANGLE_MIN} to {bike.MAXIMUM_KNEE_ANGLE_MAX}")
    print("Recommended change: adjust saddle height so that maximum knee angle is close to middle of optimum\n")
    print(f"Your minimum knee angle is {angles[1]}.")
    print(f"The recommended range is between {bike.MINIMUM_KNEE_ANGLE_MIN} and {bike.MINIMUM_KNEE_ANGLE_MAX}.")
    print(
        "In order to increase minimum knee angle, you can try to move your cleats back or lower your saddle set back. "
        "You might need to get shorter crank arms though.\n"
    )
    print(f"Your average torso angle is {angles[2]}.")
    print(
        f"Recommended range while driving in the hoods is {bike.TORSO_HORIZONTAL_ANGLE_MIN} to "
        f"{bike.TORSO_HORIZONTAL_ANGLE_MAX}. "
    )
    print("To increase the torso angle, raise the cockpit by adding spacers or installing a higher rise stem.\n")
    print(f"The average angle between your upper arm and torso is {angles[3]}.")
    print(f"The recommended range is between {bike.SHOULDER_ANGLE_MIN} and {bike.SHOULDER_ANGLE_MAX}.")
    print(f"Your average elbow angle is {angles[4]}.")
    print(f"The recommended range is {bike.ELBOW_ANGLE_MIN} to {bike.ELBOW_ANGLE_MAX}.")
    print(
        "Shoulder and arm position can be changed by various handlebar/stem combinations. If your arm is too stretched,"
        " consider a shorter stem or one with a higher rise. In order to increase elbow angle, you can choose a longer"
        " stem or one with higher rise."
    )
