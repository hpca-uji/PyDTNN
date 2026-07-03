"""Sigmoid activation function module for PyDTNN."""

import logging

from pydtnn.activations.abstract.activation import Activation
from pydtnn.utils.constants import Array

__all__ = ("Sigmoid",)

logger = logging.getLogger(__name__)


class Sigmoid[T: Array](Activation[T]):  # noqa: D101 (generics not detected)
    """Sigmoid activation layer that squashes input values into the range (0, 1)."""
