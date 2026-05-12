"""
PyDTNN Layer base class
"""

import logging

from pydtnn.abstract.layerable import Layerable
from pydtnn.utils import find_component
from pydtnn.utils.constants import Array

__all__ = (
    "Layer",
    "LayerError",
    "ParameterException",
    "select",
)

logger = logging.getLogger(__name__)


class LayerError(ValueError):
    """Exception raised for errors occurring within a Layer."""

    pass


class ParameterException(LayerError):
    """Exception raised for invalid layer parameters."""

    pass


class Layer[T: Array](Layerable[T]):
    """Base class for all neural network layers in PyDTNN."""

    pass


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
