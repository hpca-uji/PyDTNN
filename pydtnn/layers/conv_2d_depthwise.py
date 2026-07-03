"""Depthwise 2D convolution layer implementation for the PyDTNN framework."""

import logging

from pydtnn.layers.abstract.conv_2d import AbstractConv2D
from pydtnn.utils.constants import Array

__all__ = ("Conv2DDepthwise",)

logger = logging.getLogger(__name__)


class Conv2DDepthwise[T: Array](AbstractConv2D[T]):  # noqa: D101 (generics not detected)
    """A 2D depthwise convolution layer that applies a single filter per input channel."""
