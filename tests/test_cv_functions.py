#!/usr/bin/env python
"""
    test_cv_functions.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 23/06/2025

    Version: 0.1

    Description:
        Tests for cv functions library

    Change History:
        0.1: Created.
"""
import numpy as np
import cv2
from lab_monitor.cv_functions import BarrelUndistortTransform


def test_barrel_undistort_transform_apply():
    """        
    Tests the BarrelUndistortTransform's apply method.
    """
    # Create a dummy image (e.g. a black square with a white grid)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.line(img, (50, 0), (50, 199), (255, 255, 255), 1)
    cv2.line(img, (0, 100), (199, 100), (255, 255, 255), 1)

    # Initialize transform with known values
    k1 = -0.2
    k2 = 0.01
    transform = BarrelUndistortTransform(image_shape=img.shape, k1=k1, k2=k2)

    # Apply distortion correction
    result = transform.apply(img)

    # Check that output has same shape and dtype
    assert result.shape == img.shape
    assert result.dtype == img.dtype

    # Optional: verify that some distortion occurred (non-equality)
    assert not np.array_equal(img, result)
