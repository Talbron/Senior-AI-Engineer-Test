#!/usr/bin/env python
"""
    test_dino_functions.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 24/06/2025

    Version: 0.1

    Description:
        Tests for dino functions library

    Change History:
        0.1: Created.
"""
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import torch
from lab_monitor.dino_functions import DinoProcess


@pytest.fixture
def dummy_cv_image():
    """
        Creates a dummy OpenCV image for testing.
    Returns:
        np.ndarray: A dummy BGR image of shape (100, 100, 3).
    """
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@patch("lab_monitor.dino_functions.load_model")
def test_init_sets_defaults(mock_load_model):
    """
        Tests that DinoProcess initializes with default values.
    Args:
        mock_load_model (MagicMock): Mock for the load_model function.
    """
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    dp = DinoProcess()

    assert dp.device in ("cuda", "cpu")
    assert dp.model == mock_model
    assert "glass bottle" in dp.text_prompt


@patch("lab_monitor.dino_functions.T.Compose")
@patch("lab_monitor.dino_functions.cv2.cvtColor")
@patch("lab_monitor.dino_functions.Image.fromarray")
def test_transform_calls_pipeline(mock_fromarray, mock_cvtColor, mock_compose, dummy_cv_image):
    """
        Tests that the _transform method applies the correct transformations.
    Args:
        mock_fromarray (MagicMock): Mock for PIL Image.fromarray.
        mock_cvtColor (MagicMock): Mock for cv2.cvtColor.
        mock_compose (MagicMock): Mock for the T.Compose transformation pipeline.
        dummy_cv_image (np.ndarray): A dummy OpenCV image.
    """
    mock_tensor = torch.rand(3, 224, 224)
    transform_fn = MagicMock(return_value=(mock_tensor, None))
    mock_compose.return_value = transform_fn
    mock_fromarray.return_value = "PIL_IMAGE"

    dp = DinoProcess()
    transformed = dp._transform(dummy_cv_image)  # pylint: disable=W0212

    mock_cvtColor.assert_called_once()
    mock_fromarray.assert_called_once()
    transform_fn.assert_called_once_with("PIL_IMAGE", None)
    assert isinstance(transformed, torch.Tensor)


@patch("lab_monitor.dino_functions.predict")
@patch.object(DinoProcess, "_transform")
def test_process_image_calls_predict(mock_transform, mock_predict, dummy_cv_image):
    """
        Tests that process_image calls the transform and predict methods correctly.
    Args:
        mock_transform (MagicMock): Mock for the _transform method.
        mock_predict (MagicMock): Mock for the predict function.
        dummy_cv_image (np.ndarray): A dummy OpenCV image.
    """
    mock_transform.return_value = torch.rand(3, 224, 224)
    dummy_boxes = np.array([[0, 0, 10, 10]])
    dummy_logits = np.array([0.9])
    dummy_phrases = ["glass bottle"]
    mock_predict.return_value = (dummy_boxes, dummy_logits, dummy_phrases)

    dp = DinoProcess()
    boxes, logits, phrases = dp.process_image(dummy_cv_image)

    mock_transform.assert_called_once()
    mock_predict.assert_called_once()
    assert isinstance(boxes, np.ndarray)
    assert isinstance(logits, np.ndarray)
    assert isinstance(phrases, list)


@patch("lab_monitor.dino_functions.annotate")
def test_annotate_image_calls_annotate(mock_annotate, dummy_cv_image):
    """
        Tests that annotate_image calls the annotate function with correct parameters.
    Args:
        mock_annotate (MagicMock): Mock for the annotate function.
        dummy_cv_image (np.ndarray): A dummy OpenCV image.
    """
    dummy_boxes = np.array([[0, 0, 10, 10]])
    dummy_logits = np.array([0.9])
    dummy_phrases = ["bottle"]
    mock_annotate.return_value = dummy_cv_image

    dp = DinoProcess()
    result = dp.annotate_image(dummy_cv_image, dummy_boxes, dummy_logits, dummy_phrases)

    mock_annotate.assert_called_once_with(
        image_source=dummy_cv_image,
        boxes=dummy_boxes,
        logits=dummy_logits,
        phrases=dummy_phrases
    )
    assert np.array_equal(result, dummy_cv_image)
