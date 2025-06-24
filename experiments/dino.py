import torch
import os
import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate


OUTPUT_DIR = "../samples/detected"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEXT_PROMPT = "glass bottle, blue bottle cap, glass petri dish, empty petri dish, hand, circular glass dish"
IMAGE_NAME = "filling.png"
IMAGE_PATH = os.path.join("../samples/corrected/", IMAGE_NAME)
OUTPUT_PATH = os.path.join("../samples/detected/", IMAGE_NAME)
TEXT_PROMPT = "glass bottle, blue bottle cap, glass petri dish, empty petri dish, hand, circular glass dish"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(
    model_config_path="/workspaces/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    model_checkpoint_path="../weights/groundingdino_swint_ogc.pth",
    device=device
)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite(OUTPUT_PATH, annotated_frame)
