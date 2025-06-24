#!/usr/bin/env python
"""
    cv_functions.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 23/06/2025

    Version: 0.1

    Description:
        Contains a library of OpenCV Image transforms for use in pipeline

    Change History:
        0.1: Created.
"""
import cv2
import numpy as np


class BarrelUndistortTransform:
    """
        Applies barrel distortion correction using predefined k1 and k2 values.
        We default to some predefined values we determined experimentally.
        Some very rough estimates used for camera matrix
    Args:
        image_shape (tuple): Shape of input frames (height, width).
        k1 (float): Radial distortion coefficient k1.
        k2 (float): Radial distortion coefficient k2.
        camera_matrix (np.ndarray): The camera intrinsic matrix.
        dist_coeffs (np.ndarray): The distortion coefficients array.
    """
    def __init__(self, image_shape, k1: float = -0.282, k2: float = -0.282):
        h, w = image_shape[:2]
        self.k1 = k1
        self.k2 = k2
        focal = max(w, h)
        cx, cy = w / 2, h / 2

        self.camera_matrix = np.array([[focal, 0, cx],
                                       [0, focal, cy],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
            Undistorts a given frame using stored distortion parameters.
        Args:
            frame (np.ndarray): Input distorted image/frame.

        Returns:
            np.ndarray: Undistorted output image.
        """
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
