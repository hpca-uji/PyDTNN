"""Cython implementation of the 2D convolutional layer."""

import logging

from pydtnn.backends.cython.layers.abstract.conv_2d import AbstractConv2DCython
from pydtnn.backends.numpy.layers.conv_2d import Conv2DNumpy

"""Cython implementation of the 2D convolutional layer."""

__all__ = ("Conv2DCython",)

logger = logging.getLogger(__name__)


class Conv2DCython(Conv2DNumpy, AbstractConv2DCython):
    """
    2D convolutional layer using Cython for optimized performance.
    
    Inherits from Conv2DNumpy for high-level logic and AbstractConv2DCython 
    for Cython-specific interface definitions.
    """
    ...