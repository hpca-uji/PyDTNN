"""Multiplication layer module for PyDTNN."""

import logging

from pydtnn.layers.abstract.layer import Layer
from pydtnn.utils.constants import Array

__all__ = ("Multiplication",)

logger = logging.getLogger(__name__)


class Multiplication[T: Array](Layer[T]):  # noqa: D101 (generics not detected)
    """Layer that performs element-wise multiplication of input tensors."""
