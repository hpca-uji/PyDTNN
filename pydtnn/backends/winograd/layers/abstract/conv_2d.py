"""
Winograd-based 2D convolution abstract layer implementation.
"""

import logging

from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.backends.winograd.layers.layer import LayerWinograd

__all__ = ("AbstractConv2DWinograd",)

logger = logging.getLogger(__name__)


class AbstractConv2DWinograd(AbstractConv2DNumpy, LayerWinograd):
    """
    Abstract base class for 2D convolution layers utilizing the Winograd algorithm.
    """

    ...
