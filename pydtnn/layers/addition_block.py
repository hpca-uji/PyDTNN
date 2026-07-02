"""Module for the AdditionBlock layer in the PyDTNN framework."""

import logging

from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.utils.constants import Array

__all__ = ("AdditionBlock",)

logger = logging.getLogger(__name__)


class AdditionBlock[T: Array](AbstractBlockLayer[T]):
    """A layer that performs element-wise addition of input tensors."""

    pass
