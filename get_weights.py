#!/usr/bin/env python
"""
    get_weights.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 23/06/2025

    Version: 0.1

    Description:
        A script to download model weights for image segmentation.
        This script downloads the necessary weights for GroundingDINO and SAM
        and saves them in a local directory called 'weights'.

    Change History:
        0.1: Created.
"""
import os
import requests
from tqdm import tqdm

os.makedirs("weights", exist_ok=True)

weights = {
    "groundingdino_swint_ogc.pth":
        "https://github.com/IDEA-Research/GroundingDINO/"
        "releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "sam_vit_h_4b8939.pth":
        "https://dl.fbaipublicfiles.com/segment_anything/"
        "sam_vit_h_4b8939.pth"
}

for filename, url in weights.items():
    out_path = os.path.join("weights", filename)
    if os.path.exists(out_path):
        print(f"{filename} already exists, skipping.")
        continue

    print(f"Downloading {filename}...")

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(out_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"Saved to {out_path}")
    else:
        print(f"Failed to download {filename} (HTTP {response.status_code})")
