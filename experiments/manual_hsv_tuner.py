#!/usr/bin/env python
"""
    manual_hsv_tuner.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 23/06/2025

    Version: 0.1

    Description:
        A quick experiment to deterine hsv threshold values for an set of filters

    Change History:
        0.1: Created.
"""
import cv2
import numpy as np


class HSVRangeTuner:
    """
    Interactive application to adjust HSV threshold values using trackbars.
    Displays original HSV image and a filtered mask using the selected HSV range.
    """

    def __init__(self, image_path: str):
        """
            Initialize the HSV range tuner with an image.
            Args:
                image_path (str): Path to the input image file.
        """
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        self.window_mask = 'HSV Mask'
        self.window_hsv = 'HSV Image'
        cv2.namedWindow(self.window_mask)
        cv2.namedWindow(self.window_hsv)

        self._create_trackbars()

    def null(self, x):
        """
        Dummy callback function for trackbar events.

        Args:
            x: Trackbar position (ignored).
        """
        pass

    def _create_trackbars(self):
        """
            Create trackbars for adjusting HSV thresholds.
        """
        for win in [self.window_mask]:
            cv2.createTrackbar('H min', win, 0, 179, self.null)
            cv2.createTrackbar('H max', win, 179, 179, self.null)
            cv2.createTrackbar('S min', win, 0, 255, self.null)
            cv2.createTrackbar('S max', win, 255, 255, self.null)
            cv2.createTrackbar('V min', win, 0, 255, self.null)
            cv2.createTrackbar('V max', win, 255, 255, self.null)

    def run(self):
        """
            Run the HSV range tuning application.
        """
        while True:
            h_min = cv2.getTrackbarPos('H min', self.window_mask)
            h_max = cv2.getTrackbarPos('H max', self.window_mask)
            s_min = cv2.getTrackbarPos('S min', self.window_mask)
            s_max = cv2.getTrackbarPos('S max', self.window_mask)
            v_min = cv2.getTrackbarPos('V min', self.window_mask)
            v_max = cv2.getTrackbarPos('V max', self.window_mask)

            lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

            mask = cv2.inRange(self.img_hsv, lower, upper)

            cv2.imshow(self.window_hsv, self.img_bgr)
            cv2.imshow(self.window_mask, mask)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    tuner = HSVRangeTuner('../samples/corrected/filled1.png')  # Replace with your image path
    tuner.run()
