"""
PyDTNN datasets module.

This module provides utilities for managing and selecting dataset implementations
within the PyDTNN framework.
"""

from pydtnn.datasets.abstract import Dataset
from pydtnn.utils import find_component


def select(name: str) -> type[Dataset]:
    """
    Select a dataset class by name.

    This function dynamically imports and returns a dataset class based on its
    string name. It searches within the current package for the specified class.

    Args:
        name: The string name of the dataset class to select.

    Returns:
        The dataset class type.

    Raises:
        AssertionError: If the package context cannot be determined.
    """
    assert __package__, "Package not found!"
    return find_component(__package__, name)
