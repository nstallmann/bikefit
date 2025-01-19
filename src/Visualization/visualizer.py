import os
from math import inf
from typing import Optional

import matplotlib.axes
import mediapipe
import numpy as np
from matplotlib import pyplot as plt

from Constants.bike import RoadBike, MountainBike
from Constants.constants import PathConstants
from Dataclasses.body_angles import BodyAngles
from Dataclasses.body_position import BodyPosition
from Dataclasses.frame import Frame
from Visualization.drawing_style import ColorStyles, DrawingStyles


class Visualizer:
    def __init__(
        self,
        frame: Optional[Frame] = None,
        body_position: Optional[BodyPosition] = None,
        body_angles: Optional[BodyAngles] = None,
    ):
        self.frame = frame
        self.body_position = body_position
        self.body_angles = body_angles

    def draw_landmarks_on_image_and_annotate(self):
        np_image = self.frame.image.numpy_view()

        plt.imshow(np_image)
        self._connect_body_landmarks()
        self._add_landmark_points()
        self._add_angle_info()
        plt.axis("off")
        plt.savefig(self.frame.path_to_image, bbox_inches="tight", pad_inches=0)
        plt.close()

    @staticmethod
    def save_image_of_top_and_bottom_pedal_stroke(
        top_frame: Frame, bottom_frame: Frame, output_dir_name: os.PathLike | str
    ) -> None:
        np_image_left = mediapipe.Image.create_from_file(top_frame.path_to_image).numpy_view()
        np_image_right = mediapipe.Image.create_from_file(bottom_frame.path_to_image).numpy_view()
        concatenated_image = np.concatenate((np_image_left, np_image_right), axis=1)
        plt.imsave(os.path.join(output_dir_name, PathConstants.TOP_BOT_PEDAL_STROKE_FILE_NAME), concatenated_image)

    @staticmethod
    def save_image_of_angle_recommendations(
        bike: RoadBike | MountainBike,
        min_knee_angle: float,
        max_knee_angle: float,
        avg_torso_angle: float,
        avg_shoulder_angle: float,
        avg_elbow_angle: float,
        output_dir_name: os.PathLike | str,
    ) -> None:
        fig, axs = plt.subplots(nrows=5)
        fig.tight_layout()
        for ax in axs:
            Visualizer()._set_ax_params(ax)

        Visualizer._fill_ax(
            ax=axs[0],
            title="Maximum Knee Angle",
            left_border=bike.MAXIMUM_KNEE_ANGLE_MIN,
            right_border=bike.MAXIMUM_KNEE_ANGLE_MAX,
            measured_angle=max_knee_angle,
        )
        Visualizer._fill_ax(
            ax=axs[1],
            title="Minimum Knee Angle",
            left_border=bike.MINIMUM_KNEE_ANGLE_MIN,
            right_border=bike.MINIMUM_KNEE_ANGLE_MAX,
            measured_angle=min_knee_angle,
        )
        Visualizer._fill_ax(
            ax=axs[2],
            title="Average Torso Angle",
            left_border=bike.TORSO_HORIZONTAL_ANGLE_MIN,
            right_border=bike.TORSO_HORIZONTAL_ANGLE_MAX,
            measured_angle=avg_torso_angle,
        )
        Visualizer._fill_ax(
            ax=axs[3],
            title="Average Shoulder Angle",
            left_border=bike.SHOULDER_ANGLE_MIN,
            right_border=bike.SHOULDER_ANGLE_MAX,
            measured_angle=avg_shoulder_angle,
        )
        Visualizer._fill_ax(
            ax=axs[4],
            title="Average Elbow Angle",
            left_border=bike.ELBOW_ANGLE_MIN,
            right_border=bike.ELBOW_ANGLE_MAX,
            measured_angle=avg_elbow_angle,
        )
        plt.savefig(
            os.path.join(output_dir_name, PathConstants.MEASURED_VS_RECOMMENDED_ANGLES_FILE_NAME), bbox_inches="tight"
        )

    # region Annotate Frames Helper Functions
    def _connect_body_landmarks(self):
        self._draw_line(self.body_position.wrist_position_px, self.body_position.elbow_position_px)
        self._draw_line(self.body_position.elbow_position_px, self.body_position.shoulder_position_px)
        self._draw_line(self.body_position.shoulder_position_px, self.body_position.hip_position_px)
        self._draw_line(self.body_position.hip_position_px, self.body_position.knee_position_px)
        self._draw_line(self.body_position.knee_position_px, self.body_position.ankle_position_px)

    def _add_landmark_points(self):
        self._draw_point(self.body_position.wrist_position_px)
        self._draw_point(self.body_position.elbow_position_px)
        self._draw_point(self.body_position.shoulder_position_px)
        self._draw_point(self.body_position.hip_position_px)
        self._draw_point(self.body_position.knee_position_px)
        self._draw_point(self.body_position.ankle_position_px)

    def _add_angle_info(self):
        plt.text(
            x=10,
            y=10,
            s=f"knee angle: {round(self.body_angles.knee_angle, 1)}°\n"
            f"torso angle: {round(self.body_angles.torso_angle, 1)}°\n"
            f"shoulder angle:{round(self.body_angles.shoulder_angle, 1)}°\n"
            f"elbow angle:{round(self.body_angles.elbow_angle, 1)}°\n",
            fontsize=DrawingStyles.PLT_FONT_SIZE,
            ha="left",
            va="top",
        )

    @staticmethod
    def _draw_line(
        start_point: np.ndarray,
        end_point: np.ndarray,
    ):
        plt.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            color=ColorStyles.PLT_LINE_COLOR,
            linewidth=DrawingStyles.THICKNESS_LINES,
            zorder=DrawingStyles.Z_ORDER_LINES,
        )

    @staticmethod
    def _draw_point(coordinates: np.ndarray):
        plt.scatter(
            [coordinates[0]],
            [coordinates[1]],
            color=ColorStyles.PLT_POINT_COLOR,
            edgecolors=ColorStyles.PLT_LINE_COLOR,
            linewidths=DrawingStyles.THICKNESS_POINT_BORDER,
            s=DrawingStyles.THICKNESS_POSE_LANDMARKS,
            zorder=DrawingStyles.Z_ORDER_POINTS,
        )

    # endregion Annotate Frames Helper Functions

    # region Angle Recommendations Helper Functions
    @staticmethod
    def _set_ax_params(ax: matplotlib.axes.Axes) -> None:
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

    @staticmethod
    def _fill_ax(
        ax: matplotlib.axes.Axes,
        title: str,
        left_border: float,
        right_border: float,
        measured_angle: float,
    ) -> None:
        gradient = np.linspace(left_border, right_border, ColorStyles.GRADIENT_STEP_SIZE)
        gradient = np.vstack((gradient, gradient))
        ax.imshow(gradient, aspect="auto", cmap=ColorStyles.CMAP_RGR)
        measured_angle_plot = max(measured_angle, left_border)
        measured_angle_plot = min(measured_angle_plot, right_border)
        line = (measured_angle_plot - left_border) * (ColorStyles.GRADIENT_STEP_SIZE - 1) / (right_border - left_border)
        ax.vlines(line, ymin=0, ymax=1, color="black", linewidth=4)
        ax.set_ylim([0, 1])
        Visualizer._place_text_to_ax(ax=ax, x=-0.01, y=0.5, text=f"{left_border}°", ha="right")
        Visualizer._place_text_to_ax(ax=ax, x=1.01, y=0.5, text=f"{right_border}°", ha="left")
        ax_title_text = ax.set_title(label=title + f" {round(measured_angle, 1)}°")

    @staticmethod
    def _place_text_to_ax(
        ax: matplotlib.axes.Axes,
        x: float,
        y: float,
        text: str,
        ha: str = "left",
        color: tuple[float, float, float] = ColorStyles.PLT_TITLE_MAIN_COLOR,
    ):
        ax_text = ax.text(
            x=x,
            y=y,
            s=text,
            color=color,
            horizontalalignment=ha,
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax_text.set_color(color)

    @staticmethod
    def _get_title_end_coordinates(title_box_boundaries) -> tuple:
        box_end_point_x, box_end_point_y = -inf, 0
        for x, y in title_box_boundaries:
            box_end_point_y += y / 4
            box_end_point_x = max(box_end_point_x, x)
        return box_end_point_x + 0.01, box_end_point_y

    # endregion Angle Recommendations Helper Functions
