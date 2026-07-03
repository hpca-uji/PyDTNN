"""2D Max Pooling layer implementation for the PyDTNN framework."""

import logging

from pydtnn.layers.abstract.pool_2d_layer import AbstractPool2DLayer
from pydtnn.utils.constants import Array

__all__ = ("MaxPool2D",)

logger = logging.getLogger(__name__)


class MaxPool2D[T: Array](AbstractPool2DLayer[T]):  # noqa: D101 (generics not detected)
    """Performs 2D max pooling on the input tensor."""
