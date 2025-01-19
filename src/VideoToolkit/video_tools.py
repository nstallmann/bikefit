#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
from typing import Optional

import cv2
import numpy as np

from Constants.constants import PathConstants


class VideoTools:
    def __init__(self):
        self.run_folder_path: Optional[str | os.PathLike] = None
        self.frames_folder_path: Optional[str | os.PathLike] = None

    def extract_images(self):
        self._create_run_folder()

        if len(os.listdir(self.frames_folder_path)) > 0:
            return

        video_capture = cv2.VideoCapture(PathConstants.INPUT_FILE_PATH)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        count = 0
        print("Converting video..\n")
        # Start converting video
        while video_capture.isOpened():
            ret, image = video_capture.read()
            if not ret:
                continue
            cv2.imwrite(os.path.join(self.frames_folder_path, PathConstants.FRAME_FILE_NAME.format(count)), image)
            count = count + 1
            if count > (video_length - 1):
                video_capture.release()
                print(f"Done extracting frames. {count} frames extracted")
                break

    def _create_gif_from_frames(self, frames_list: list[np.ndarray]):
        raise NotImplementedError

    def _create_run_folder(self):
        run_folders = glob.glob(os.path.join(PathConstants.OUTPUT_DIR_PATH, f"{PathConstants.RUN_DIR_BASE_NAME}_*"))
        run_numbers = [int(folder.split("_")[1]) for folder in run_folders]

        next_run_number = max(run_numbers, default=0) + 1
        self.run_folder_path = os.path.join(
            PathConstants.OUTPUT_DIR_PATH, f"{PathConstants.RUN_DIR_BASE_NAME}_{next_run_number}"
        )
        self.frames_folder_path = os.path.join(self.run_folder_path, PathConstants.FRAMES_DIR_NAME)
        os.makedirs(self.frames_folder_path, exist_ok=False)

    def get_frames_folder_path(self):
        return self.frames_folder_path
