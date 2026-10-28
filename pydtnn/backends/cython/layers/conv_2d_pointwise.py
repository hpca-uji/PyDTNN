"""Cython implementation of 2D pointwise convolutional layers."""

import logging

from pydtnn.backends.cython.layers.abstract.conv_2d import AbstractConv2DCython
from pydtnn.backends.numpy.layers.conv_2d_pointwise import Conv2DPointwiseNumpy

__all__ = ("Conv2DPointwiseCython",)

logger = logging.getLogger(__name__)


class Conv2DPointwiseCython(Conv2DPointwiseNumpy, AbstractConv2DCython):
    """2D pointwise convolutional layer using Cython kernels for acceleration."""
