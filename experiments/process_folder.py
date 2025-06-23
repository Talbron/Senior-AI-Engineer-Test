#!/usr/bin/env python
"""
    process_folder.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 23/06/2025

    Version: 0.1

    Description:
        A quick experiment to transform sample images

    Change History:
        0.1: Created.
"""
import os
import cv2
from tqdm import tqdm
from lab_monitor.cv_functions import BarrelUndistortTransform


def process_folder(input_folder, output_folder, k1, k2):
    os.makedirs(output_folder, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        img = cv2.imread(input_path)
        if img is None:
            print(f"Skipped (cannot load): {filename}")
            continue

        transformer = BarrelUndistortTransform(img.shape, k1, k2)
        corrected = transformer.apply(img)
        cv2.imwrite(output_path, corrected)


if __name__ == "__main__":
    input_dir = "../samples/raw"
    output_dir = "../samples/corrected"
    k1 = -0.282
    k2 = -0.282
    process_folder(input_dir, output_dir, k1, k2)
