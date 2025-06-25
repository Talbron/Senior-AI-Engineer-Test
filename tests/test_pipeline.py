from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from lab_monitor.pipeline import process_video


@patch("lab_monitor.pipeline.cv2.VideoCapture")
@patch("lab_monitor.pipeline.cv2.VideoWriter")
@patch("lab_monitor.pipeline.BarrelUndistortTransform")
@patch("lab_monitor.pipeline.DinoProcess")
@patch("lab_monitor.pipeline.OverlapEventTracker")
def test_process_video_success(mock_tracker, mock_dino, mock_transform, mock_writer, mock_capture):
    """
        Test the process_video function with mocked components to ensure it processes a video correctly.
    Args:
        mock_tracker: Mock for OverlapEventTracker.
        mock_dino: Mock for DinoProcess.
        mock_transform: Mock for BarrelUndistortTransform.
        mock_writer: Mock for cv2.VideoWriter.
        mock_capture: Mock for cv2.VideoCapture.
    """
    mock_cap = MagicMock()
    mock_cap.read.side_effect = [(True, np.ones((480, 640, 3), dtype=np.uint8))] * 3 + [(False, None)]
    mock_cap.get.side_effect = lambda x: {cv2.CAP_PROP_FRAME_COUNT: 3, cv2.CAP_PROP_FPS: 30}[x]
    mock_capture.return_value = mock_cap

    mock_writer_instance = MagicMock()
    mock_writer.return_value = mock_writer_instance

    mock_transform_instance = MagicMock()
    mock_transform.return_value = mock_transform_instance
    mock_transform_instance.apply.return_value = np.ones((480, 640, 3), dtype=np.uint8)

    mock_dino_instance = MagicMock()
    mock_dino.return_value = mock_dino_instance
    mock_dino_instance.process_image.return_value = ([np.array([0, 0, 1, 1])], [0.9], ["label"])
    mock_dino_instance.annotate_image.return_value = np.ones((480, 640, 3), dtype=np.uint8)
    mock_dino_instance.map_label.return_value = "MappedLabel"

    progress_updates = []
    process_video("input.mp4", "output.mp4", "log.csv", progress_callback=progress_updates.append)

    assert progress_updates[-1] == 100
    assert mock_writer_instance.write.call_count == 3
    assert mock_tracker.return_value.update.call_count == 3
    mock_tracker.return_value.close.assert_called_once()
