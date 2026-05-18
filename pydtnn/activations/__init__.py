"""
Activation functions module for PyDTNN.

This module provides utilities for managing and retrieving activation function
implementations within the PyDTNN framework.
"""

from pydtnn.activations.abstract.activation import Activation
from pydtnn.utils import find_component


def select(name: str) -> type[Activation]:
    """
    Retrieves an activation class by its name from the package.

    Args:
        name: The name of the activation component to retrieve.

    Returns:
        The class type of the requested activation.
    """
    assert __package__, "Package not found!"
    return find_component(__package__, name)
