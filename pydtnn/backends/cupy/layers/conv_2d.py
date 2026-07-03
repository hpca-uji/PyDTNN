"""CuPy implementation of 2D Convolutional layers for the PyDTNN framework."""

import logging

from pydtnn.backends.cupy.layers.abstract.conv_2d import AbstractConv2DCupy
from pydtnn.backends.cupy.layers.abstract.layer import LayerCupy
from pydtnn.backends.numpy.layers.conv_2d import Conv2DNumpy

__all__ = ("Conv2DCupy",)

logger = logging.getLogger(__name__)


class Conv2DCupy(Conv2DNumpy, AbstractConv2DCupy, LayerCupy):
    """2D Convolutional layer implementation using CuPy for GPU acceleration."""
