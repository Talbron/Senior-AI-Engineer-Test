#!/usr/bin/env python
"""
    test_event_tracker.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 23/06/2025

    Version: 0.1

    Description:
        Tests for event tracker functionality.

    Change History:
        0.1: Created.
"""
import tempfile
import os
import pytest
from lab_monitor.event_tracker import OverlapEventTracker


@pytest.fixture
def temp_log_file():
    """
        Fixture to create a temporary log file for testing.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tf:
        path = tf.name
    yield path
    os.remove(path)


def test_boxes_overlap():
    """
        Tests the box overlap detection logic.
    """
    tracker = OverlapEventTracker(log_path="/dev/null", fps=30.0)

    # Fully overlapping boxes
    assert tracker.boxes_overlap((10, 10, 50, 50), (20, 20, 40, 40)) is True
    # Touching at edge (no area overlap)
    assert tracker.boxes_overlap((10, 10, 20, 20), (20, 20, 30, 30)) is False
    # Completely disjoint
    assert tracker.boxes_overlap((0, 0, 10, 10), (20, 20, 30, 30)) is False


def test_overlap_logging(temp_log_file):
    """
        Tests logging of overlap events between hand and petri dish.
        This simulates a scenario where the hand touches the petri dish and then releases it.
        The log should contain the correct actions with timestamps.
    Args:
        temp_log_file (str): Path to the temporary log file created by the fixture.
    """
    fps = 10.0
    tracker = OverlapEventTracker(log_path=temp_log_file, fps=fps)

    # Frame 0: hand and petri dish overlap
    tracker.update(0, {
        "hand": [(0, 0, 10, 10)],
        "petri dish": [(5, 5, 15, 15)]
    })

    # Frame 1: no more overlap
    tracker.update(1, {
        "hand": [(0, 0, 4, 4)],
        "petri dish": [(5, 5, 15, 15)]
    })

    tracker.close()

    with open(temp_log_file, "r") as f:
        lines = f.readlines()

    assert lines[0].strip() == "frame,timestamp,action"
    assert lines[1].strip() == "0,0.000,hand touches petri dish"
    assert lines[2].strip() == "1,0.100,hand releases petri dish"


def test_multiple_overlaps(temp_log_file):
    """
        Tests logging of multiple overlaps in a single frame.
        This simulates a scenario where the hand touches the petri dish and places a bottle cap on a bottle.
        The log should contain the correct actions with timestamps.
    Args:
        temp_log_file (str): Path to the temporary log file created by the fixture.
    """
    tracker = OverlapEventTracker(log_path=temp_log_file, fps=5.0)

    # Frame 0: two overlaps start
    tracker.update(0, {
        "hand": [(0, 0, 10, 10)],
        "petri dish": [(5, 5, 15, 15)],
        "bottle": [(20, 20, 30, 30)],
        "bottle cap": [(25, 25, 35, 35)]
    })

    # Frame 1: only one remains
    tracker.update(1, {
        "hand": [(0, 0, 10, 10)],
        "petri dish": [(5, 5, 15, 15)],
        "bottle": [(20, 20, 30, 30)],
        # bottle cap moved away
        "bottle cap": [(50, 50, 60, 60)]
    })

    tracker.close()

    with open(temp_log_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    assert "0,0.000,hand touches petri dish" in lines
    assert "0,0.000,bottle cap is placed on bottle" in lines
    assert "1,0.200,bottle cap is removed from bottle" in lines


def test_no_log_for_untracked_pair(temp_log_file):
    """
        Tests that no log entry is created for pairs that are not in pairs_to_track.
        This simulates a scenario where the hand overlaps with an untracked object (apple).
        The log should only contain the header and no action entries.
    Args:
        temp_log_file (str): Path to the temporary log file created by the fixture.
    """
    tracker = OverlapEventTracker(log_path=temp_log_file, fps=10.0)

    # Frame 0: hand overlaps with an untracked object
    tracker.update(0, {
        "hand": [(0, 0, 10, 10)],
        "apple": [(5, 5, 15, 15)]  # not in pairs_to_track
    })

    tracker.close()

    with open(temp_log_file, "r") as f:
        lines = f.readlines()

    # Should only contain header
    assert len(lines) == 1
    assert lines[0].startswith("frame,timestamp,action")
