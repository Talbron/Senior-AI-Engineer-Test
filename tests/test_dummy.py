#!/usr/bin/env python
"""
    test_dummy.py:
    quick dummy test to check pipeline.
"""
from lab_monitor.functions import DummyClass


def test_example():
    dummy = DummyClass()
    assert dummy.get_val() == 1
    dummy.set_val(val=2)
    assert dummy.val == 2
