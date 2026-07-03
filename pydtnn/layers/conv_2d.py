"""2D Convolutional layer implementation for the PyDTNN framework."""

import logging

from pydtnn.layers.abstract.conv_2d import AbstractConv2D
from pydtnn.utils.constants import Array

__all__ = ("Conv2D",)

logger = logging.getLogger(__name__)


class Conv2D[T: Array](AbstractConv2D[T]):  # noqa: D101 (generics not detected)
    """Standard 2D convolutional layer that performs cross-correlation over input tensors."""
