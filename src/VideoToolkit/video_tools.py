#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from pathlib import Path


def extract_images(path_in, path_out):
    # Create frame folder
    Path(path_out).mkdir(parents=True, exist_ok=True)
    
    # Start capturing the feed
    vidcap = cv2.VideoCapture(path_in)
    # Find number of frames
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    count = 0
    print ("Converting video..\n")
    # Start converting video
    while vidcap.isOpened():       
        ret, image = vidcap.read()
        if not ret:
            continue
        # Store extracted image in its folder
        cv2.imwrite(path_out + f"/frame{count:04d}.jpg", image)     # save frame as JPEG file
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Release the feed
            vidcap.release()
            print ("Done extracting frames.\n%d frames extracted" % count)
            break

