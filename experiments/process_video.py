#!/usr/bin/env python
"""
    calibrate_video.py:

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
from tqdm import tqdm
from lab_monitor.cv_functions import BarrelUndistortTransform
from lab_monitor.dino_functions import DinoProcess
from lab_monitor.event_tracker import OverlapEventTracker


video_path = "../data/AICandidateTest-FINAL.mp4"
output_path = "../data/AICandidateTest-DETECTED.mp4"

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Failed to read the video file.")
    cap.release()
    exit()

# Initialize the undistortion transform
transform = BarrelUndistortTransform(frame.shape, k1=-0.182, k2=0.0032)

# Get original video properties
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
event_tracker = OverlapEventTracker(log_path="../data/event_log.csv", fps=fps)
frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' for .avi, 'mp4v' for .mp4

# Initialize the video writer
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

network = DinoProcess()
network.load_model()
frame_number = 0
with tqdm(total=frame_count, desc="Processing video", unit="frame") as pbar:
    while ret:
        # correct frame for barrel distortion
        undistorted = transform.apply(frame)
        # detect objects in the undistorted frame
        boxes, logits, phrases = network.process_image(undistorted)
        phrases = [network.map_label(phrase) for phrase in phrases]
        annotated = network.annotate_image(undistorted, boxes, logits, phrases)
        # track events based on detected objects
        detected_objects = {}
        for box, phrase in zip(boxes, phrases):
            detected_objects.setdefault(phrase.lower(), []).append(box)
        event_tracker.update(frame_number, detected_objects)
        # Annotate the frame with detected objects
        bgr_annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        out.write(bgr_annotated)  # Save the undistorted frame
        ret, frame = cap.read()
        frame_number += 1
        pbar.update(1)

event_tracker.close()
cap.release()
out.release()
