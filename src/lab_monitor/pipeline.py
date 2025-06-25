#!/usr/bin/env python
"""
    pipeline.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 25/06/2025

    Version: 0.1

    Description:
        A bit of redundant code that is a callable wrapper around process_video.py.

    Change History:
        0.1: Created.
"""
import cv2
from lab_monitor.cv_functions import BarrelUndistortTransform
from lab_monitor.dino_functions import DinoProcess
from lab_monitor.event_tracker import OverlapEventTracker


def process_video(video_path: str, output_path: str, log_path: str, progress_callback=None):
    """
        Process a video file and save the output.
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the processed video.
        log_path (str): Path to save the event log.
        progress_callback (callable, optional): A callback function to report progress.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Failed to read the video file.")

    transform = BarrelUndistortTransform(frame.shape, k1=-0.182, k2=0.0032)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    event_tracker = OverlapEventTracker(log_path=log_path, fps=fps)
    frame_size = (frame.shape[1], frame.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    network = DinoProcess()
    network.load_model()

    frame_number = 0
    while ret:
        undistorted = transform.apply(frame)
        boxes, logits, phrases = network.process_image(undistorted)
        phrases = [network.map_label(p) for p in phrases]
        annotated = network.annotate_image(undistorted, boxes, logits, phrases)

        detected_objects = {}
        for box, phrase in zip(boxes, phrases):
            detected_objects.setdefault(phrase.lower(), []).append(box)
        event_tracker.update(frame_number, detected_objects)

        bgr_annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        out.write(bgr_annotated)

        ret, frame = cap.read()
        frame_number += 1
        if progress_callback:
            progress_callback(int((frame_number / frame_count) * 100))

    event_tracker.close()
    cap.release()
    out.release()
