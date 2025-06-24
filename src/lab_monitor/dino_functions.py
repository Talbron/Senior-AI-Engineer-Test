#!/usr/bin/env python
"""
    dino_functions.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 24/06/2025

    Version: 0.1

    Description:
        Contains a library of Dino Image transforms for use in pipeline

    Change History:
        0.1: Created.
"""
from typing import Tuple, List
import cv2
from PIL import Image
import torch
import numpy as np
from groundingdino.util.inference import predict, annotate, load_model
import groundingdino.datasets.transforms as T


class DinoProcess:
    """
        A class to handle image processing using GroundingDINO.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", text_prompt: str = None):
        self.device = device
        self.model = None
        self.text_prompt = text_prompt or (
            "glass bottle, blue bottle cap, glass petri dish, empty petri dish, hand, circular glass dish"
        )

    def load_model(self,
                   model_config_path="/workspaces/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                   model_checkpoint_path="../weights/groundingdino_swint_ogc.pth"):
        """
            Explicitly loads the GroundingDINO model.
        Args:
            model_config_path (str): Path to the model configuration file.
            model_checkpoint_path (str): Path to the model checkpoint file.
        """
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=self.device
        )

    def _transform(self, cv_image: np.array) -> torch.Tensor:
        """
            Transforms a CV image to a tensor suitable for GroundingDINO.
        Args:
            cv_image (np.array): Input image in OpenCV format (BGR).
        Returns:
            torch.Tensor: Transformed image tensor.
        """
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(pil_image, None)
        return image_transformed

    def process_image(self, cv_image: np.array,
                      box_threshold: float = 0.35,
                      text_threshold: float = 0.25) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Processes an image using GroundingDINO to detect objects based on a text prompt.
        Args:
            cv_image (np.array): Input image in OpenCV format (BGR).
            box_threshold (float): Threshold for box detection.
            text_threshold (float): Threshold for text detection.
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Detected boxes, logits, and phrases.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        image_transformed = self._transform(cv_image)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=self.text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        return boxes, logits, phrases

    def annotate_image(self, cv_image: np.array, boxes: np.ndarray,
                       logits: np.ndarray, phrases: List[str]) -> np.ndarray:
        """
        Annotates the input image with detected boxes and phrases.
        Args:
            cv_image (np.array): Input image in OpenCV format (BGR).
            boxes (np.ndarray): Detected boxes from GroundingDINO.
            logits (np.ndarray): Logits corresponding to the boxes.
            phrases (List[str]): Phrases corresponding to the boxes.
        Returns:
            np.ndarray: Annotated image with boxes and phrases.
        """
        annotated_frame = annotate(image_source=cv_image, boxes=boxes,
                                   logits=logits, phrases=phrases)
        return annotated_frame

    GROUP_MAP = {
        "hand": "hand",
        "glass bottle": "bottle",
        "blue bottle cap": "bottle cap",
        "glass petri dish": "petri dish",
        "empty petri dish": "petri dish",
        "circular glass dish": "petri dish"
    }

    def map_label(self, phrase: str) -> str:
        """
        Maps a given phrase to a predefined group label.
        Args:
            phrase (str): The input phrase to map.
        Returns:
            str: The mapped group label or the original phrase if no mapping exists."""
        phrase = phrase.strip().lower()
        return self.GROUP_MAP.get(phrase, phrase)
