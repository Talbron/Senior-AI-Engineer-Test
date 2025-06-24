#!/usr/bin/env python
"""
    event_tracker.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 24/06/2025

    Version: 0.1

    Description:
        Rudimentary event tracker for detecting overlaps
        between pairs of objects in a video stream.

    Change History:
        0.1: Created.
"""


class OverlapEventTracker:
    """
        A class to track overlapping events between pairs of objects in a video stream.
    """
    def __init__(self, log_path, fps):
        """
        Args:
            log_path (str): Path to the log file.
            fps (float): Video frames per second to convert frame number to seconds.
        """
        self.fps = fps

        self.pairs_to_track = [
            ("hand", "petri dish"),
            ("hand", "bottle"),
            ("bottle cap", "bottle"),
            ("bottle", "petri dish")
        ]

        self.start_messages = {
            ("hand", "petri dish"): "hand touches petri dish",
            ("hand", "bottle"): "hand touches bottle",
            ("bottle cap", "bottle"): "bottle cap is placed on bottle",
            ("bottle", "petri dish"): "bottle is poured into petri dish",
        }
        self.end_messages = {
            ("hand", "petri dish"): "hand releases petri dish",
            ("hand", "bottle"): "hand releases bottle",
            ("bottle cap", "bottle"): "bottle cap is removed from bottle",
        }

        self.pairs_to_track = [(a.lower(), b.lower()) for a, b in self.pairs_to_track]
        self.current_overlaps = set()
        self.log_file = open(log_path, "w", encoding="utf-8")  # pylint: disable=R1732

        # Write CSV header
        self.log_file.write("frame,timestamp,action\n")

    def boxes_overlap(self, box1, box2) -> bool:
        """
            Checks if two bounding boxes overlap.
        Args:
            box1 (tuple): Bounding box in the format (x1, y1, x2, y2).
            box2 (tuple): Bounding box in the format (x1, y1, x2, y2).
        Returns:
            bool: True if boxes overlap, False otherwise.
        """
        x_a = max(box1[0], box2[0])
        y_a = max(box1[1], box2[1])
        x_b = min(box1[2], box2[2])
        y_b = min(box1[3], box2[3])
        inter_width = max(0, x_b - x_a)
        inter_height = max(0, y_b - y_a)
        return inter_width > 0 and inter_height > 0

    def update(self, frame_number, detected_objects) -> None:
        """
            Updates the tracker with the current frame number and detected objects.
        Args:
            frame_number (int): The current frame number in the video.
            detected_objects (dict): A dictionary where keys are object names and values are lists of bounding boxes.
                Example: {'hand': [(x1, y1, x2, y2), ...], 'bottle': [(x1, y1, x2, y2), ...]}
        Returns:
            None
        """
        new_overlaps = set()
        for (obj1, obj2) in self.pairs_to_track:
            boxes1 = detected_objects.get(obj1, [])
            boxes2 = detected_objects.get(obj2, [])
            for b1 in boxes1:
                for b2 in boxes2:
                    if self.boxes_overlap(b1, b2):
                        new_overlaps.add((obj1, obj2))
                        break

        started = new_overlaps - self.current_overlaps
        ended = self.current_overlaps - new_overlaps

        timestamp = frame_number / self.fps

        for pair in started:
            msg = self.start_messages.get(pair)
            if msg:
                self.log_file.write(f"{frame_number},{timestamp:.3f},{msg}\n")
        for pair in ended:
            msg = self.end_messages.get(pair)
            if msg:
                self.log_file.write(f"{frame_number},{timestamp:.3f},{msg}\n")

        self.log_file.flush()
        self.current_overlaps = new_overlaps

    def close(self):
        """
        Closes the log file.
        """
        self.log_file.close()
