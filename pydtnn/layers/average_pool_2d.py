"""
Module for 2D average pooling layer implementation.
"""
import logging

from pydtnn.layers.abstract.pool_2d_layer import AbstractPool2DLayer
from pydtnn.utils.constants import Array

__all__ = ("AveragePool2D",)

logger = logging.getLogger(__name__)


class AveragePool2D[T: Array](AbstractPool2DLayer[T]):
    """
    2D Average Pooling layer that computes the average of values in each window.
    """
    pass