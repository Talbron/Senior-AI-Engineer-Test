#!/usr/bin/env python
"""
    manual_calibration.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 23/06/2025

    Version: 0.1

    Description:
        A quick experiment to deterine calibration coefficients for an image

    Change History:
        0.1: Created.
"""
import cv2
import numpy as np


class BarrelDistortionCorrector:
    """
    Interactive application to manually correct barrel distortion in an image
    using OpenCV trackbars for distortion coefficients k1 and k2.

    Attributes:
        img (np.ndarray): The original distorted image loaded from disk.
        h (int): Image height.
        w (int): Image width.
        center (tuple): The principal point (cx, cy), assumed at image center.
        focal_length (float): Estimated focal length used for camera matrix.
        camera_matrix (np.ndarray): The intrinsic camera matrix.
        window_name (str): Name of the OpenCV window.
    """

    def __init__(self, image_path: str):
        """
        Initialize the distortion corrector with an image.

        Args:
            image_path (str): Path to the distorted input image file.

        Raises:
            FileNotFoundError: If the image cannot be loaded from the given path.
        """
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.h, self.w = self.img.shape[:2]
        self.center = (self.w / 2, self.h / 2)
        self.focal_length = max(self.w, self.h)
        self.camera_matrix = np.array([[self.focal_length, 0, self.center[0]],
                                       [0, self.focal_length, self.center[1]],
                                       [0, 0, 1]], dtype=np.float32)

        self.window_name = 'Undistort'
        cv2.namedWindow(self.window_name)

        # Constants for scaling
        self.k1_slider_max = 4000  # (-2.0 to +2.0) scaled by 1e-3
        self.k2_slider_max = 4000  # (-1.0 to +1.0) scaled by 1e-3

        # Initialize trackbars for distortion coefficients
        cv2.createTrackbar('k1 x 1e-3', self.window_name, self.k1_slider_max // 2, self.k1_slider_max, self.null)
        cv2.createTrackbar('k2 x 1e-5', self.window_name, self.k2_slider_max // 2, self.k2_slider_max, self.null)

    def null(self, x):
        """
        Dummy callback function for trackbar events.

        Args:
            x: Trackbar position (ignored).
        """
        pass

    def undistort_image(self, k1: float, k2: float) -> np.ndarray:
        """
        Apply undistortion to the loaded image using specified distortion coefficients.

        Args:
            k1 (float): Radial distortion coefficient k1.
            k2 (float): Radial distortion coefficient k2.

        Returns:
            np.ndarray: The undistorted image.
        """
        dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
        undistorted = cv2.undistort(self.img, self.camera_matrix, dist_coeffs)
        return undistorted

    def run(self):
        """
        Main loop to display the undistorted image interactively with sliders.
        Updates the image in real-time as k1 and k2 values are changed.
        Press ESC key to exit the application.
        """
        while True:
            k1_slider = cv2.getTrackbarPos('k1 x 1e-3', self.window_name)
            k2_slider = cv2.getTrackbarPos('k2 x 1e-5', self.window_name)

            # Map slider positions to distortion coefficient values
            k1 = (k1_slider - self.k1_slider_max // 2) * 1e-3  # Now from -2.0 to +2.0
            k2 = (k2_slider - self.k2_slider_max // 2) * 1e-3  # Now from -1.0 to +1.0

            undistorted = self.undistort_image(k1, k2)

            # Overlay current distortion parameters on the image
            disp = undistorted.copy()
            text = f'k1={k1:.5f}, k2={k2:.7f}'
            cv2.putText(disp, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(self.window_name, disp)
            cv2.imshow("Original", self.img)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC key
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    corrector = BarrelDistortionCorrector('../samples/filling.png')  # Image path
    corrector.run()
