#!/usr/bin/env python
"""
    dino_zero_shot_detector.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 23/06/2025

    Version: 0.1

    Description:
        A quick experiment to run GroundingDINO for zero-shot object detection
        on a single image. This script loads a pre-trained GroundingDINO model,
        applies it to an image, and saves the annotated output with detected
        objects highlighted. Runs locally.

    Change History:
        0.1: Created.
"""
import os
import cv2
from PIL import Image
import torch
from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T


OUTPUT_DIR = "../samples/detected"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEXT_PROMPT = "glass bottle, blue bottle cap, glass petri dish, empty petri dish, hand, circular glass dish"
IMAGE_NAME = "filling.png"
IMAGE_PATH = os.path.join("../samples/corrected/", IMAGE_NAME)
OUTPUT_PATH = os.path.join("../samples/detected/", IMAGE_NAME)
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# image_source, image = load_image(IMAGE_PATH) # from groundingdino.util.inference.load_image

# here we use cv2 to read the image and convert it to PIL format
# this is necessary because the predict function expects a torch tensor
# when we load from a video feed we can't use load_image,
# and we want to have a pre-processing lens correction transform step in the pipeline
# ideally we'd use transform directly on the cv2 image, but that requires a bit more work
image = cv2.imread(IMAGE_PATH)
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
image_transformed, _ = transform(pil_image, None)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(
    model_config_path="/workspaces/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    model_checkpoint_path="../weights/groundingdino_swint_ogc.pth",
    device=device
)

boxes, logits, phrases = predict(
    model=model,
    image=image_transformed,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
