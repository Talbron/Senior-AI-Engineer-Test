#!/usr/bin/env python
"""
    test_dummy.py:
    quick dummy test to check pipeline.
"""


class DummyClass:
    """
        A dummy class to test coverarge
    """
    def __init__(self):
        self.val = 1

    def get_val(self) -> float | int:
        """
            Gets a value
        Returns:
            float | int: internal value
        """
        return self.val

    def set_val(self, val: float | int):
        """
            Sets a value
        Args:
            val (float | int): value to set
        """
        self.val = val
