"""
Pointwise 2D convolution layer implementation for the PyDTNN framework.
"""
import logging

from pydtnn.layers.abstract.conv_2d import AbstractConv2D
from pydtnn.utils.constants import Array

__all__ = ("Conv2DPointwise",)

logger = logging.getLogger(__name__)


class Conv2DPointwise[T: Array](AbstractConv2D[T]):
    """
    A 2D pointwise convolution layer that performs a 1x1 convolution across input channels.
    """
    pass