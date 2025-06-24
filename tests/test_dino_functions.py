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


def test_init_sets_defaults():
    """
        Tests that DinoProcess initializes with default values.

    """
    dp = DinoProcess()

    assert dp.device in ("cuda", "cpu")
    assert dp.model is None
    assert "glass bottle" in dp.text_prompt


@patch("lab_monitor.dino_functions.load_model")
def test_load_model_sets_model(mock_load_model):
    """
        Tests that load_model sets the model attribute correctly.
    Args:
        mock_load_model (MagicMock): Mock for the load_model function.
    """
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    dp = DinoProcess()
    dp.load_model()

    assert dp.model == mock_model
    mock_load_model.assert_called_once()


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


def test_process_image_raises_if_model_not_loaded():
    """
        Tests that process_image raises an error if the model is not loaded.
    """
    dp = DinoProcess()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        dp.process_image(cv_image=MagicMock())


@patch("lab_monitor.dino_functions.predict")
def test_process_image_runs_with_model(mock_predict):
    """
        Tests that process_image calls the predict function with correct parameters.
    Args:
        mock_predict (MagicMock): Mock for the predict function.
    """
    mock_predict.return_value = ("boxes", "logits", ["phrase1", "phrase2"])
    mock_model = MagicMock()

    dp = DinoProcess()
    dp.model = mock_model
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = dp.process_image(cv_image=dummy_image)

    assert result == ("boxes", "logits", ["phrase1", "phrase2"])
    mock_predict.assert_called_once()


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


@pytest.mark.parametrize("input_phrase, expected_output", [
    ("hand", "hand"),
    ("glass bottle", "bottle"),
    ("blue bottle cap", "bottle cap"),
    ("glass petri dish", "petri dish"),
    ("empty petri dish", "petri dish"),
    ("circular glass dish", "petri dish"),
    ("unknown item", "unknown item"),
    ("  Hand ", "hand"),
    ("GLASS BOTTLE", "bottle"),
    ("Blue Bottle Cap", "bottle cap")
])
def test_map_label(input_phrase, expected_output):
    dp = DinoProcess()
    assert dp.map_label(input_phrase) == expected_output
