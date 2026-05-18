"""
PyDTNN layers module.

This module provides utilities for managing and retrieving neural network layer
implementations within the PyDTNN framework.
"""

from pydtnn.layers.abstract.layer import Layer
from pydtnn.utils import find_component


def select(name: str) -> type[Layer]:
    """
    Retrieve a Layer class by its name from the package.

    Args:
        name: The name of the layer component to retrieve.

    Returns:
        The class type of the requested layer.
    """
    assert __package__, "Package not found!"
    return find_component(__package__, name)
