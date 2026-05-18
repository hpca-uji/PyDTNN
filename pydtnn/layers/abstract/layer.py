"""
PyDTNN Layer base class
"""

import logging

from pydtnn.abstract.layerable import Layerable
from pydtnn.utils.constants import Array

__all__ = ("Layer", "LayerError", "ParameterException")

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
