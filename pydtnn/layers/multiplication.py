"""Multiplication layer module for PyDTNN."""

import logging

from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.utils.constants import Array

__all__ = ("Multiplication",)

logger = logging.getLogger(__name__)


class Multiplication[T: Array](AbstractBlockLayer[T]):  # noqa: D101 (generics not detected)
    """Layer that performs element-wise multiplication of input tensors."""
