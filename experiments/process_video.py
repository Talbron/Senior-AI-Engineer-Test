#!/usr/bin/env python
"""
    process_video.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 23/06/2025

    Version: 0.1

    Description:
        A quick experiment to check transforms on video

    Change History:
        0.1: Created.
"""
import cv2
from lab_monitor.cv_functions import BarrelUndistortTransform


video_path = "../data/AICandidateTest-FINAL.mp4"  # Replace with your actual file path
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Failed to read the video file.")
    cap.release()
    exit()

transform = BarrelUndistortTransform(frame.shape, k1=-0.182, k2=0.0032)

while ret:
    undistorted = transform.apply(frame)
    cv2.imshow("Corrected", undistorted)
    cv2.imshow("Original", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
