import torch
import numpy as np
import cv2
import os
from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_model, predict, load_image

# Load image
OUTPUT_DIR = "../samples/detected"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEXT_PROMPT = "glass bottle, blue bottle cap, glass petri dish, empty petri dish, hand, circular glass dish"
IMAGE_NAME = "filling.png"
IMAGE_PATH = os.path.join("../samples/corrected/", IMAGE_NAME)
image_source, image = load_image(IMAGE_PATH)

# Load GroundingDINO
device = "cuda" if torch.cuda.is_available() else "cpu"
dino_model = load_model(
    model_config_path="/workspaces/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    model_checkpoint_path="../weights/groundingdino_swint_ogc.pth",
    device=device
)

# Predict boxes from text prompt
boxes, logits, phrases = predict(
    model=dino_model,
    image=image,
    caption=TEXT_PROMPT,  # use generic prompt or specific one
    box_threshold=0.35,
    text_threshold=0.25
)

# Convert boxes to numpy and transform to pixel coords
image_np = np.array(image)
h, w, _ = image_np.shape
boxes = boxes * torch.tensor([w, h, w, h])
boxes = boxes.numpy().astype(int)

# Load SAM
sam = sam_model_registry["vit_h"](checkpoint="../weights/sam_vit_h_4b8939.pth")
sam.to(device)
predictor = SamPredictor(sam)
predictor.set_image(image_source)

# Run SAM for each box
for i, box in enumerate(boxes):
    mask, _, _ = predictor.predict(box=box, multimask_output=False)
    mask = mask[0].astype(np.uint8) * 255
    # Save mask
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"mask_{i}.png")
    cv2.imwrite(f"mask_{i}.png", mask)
