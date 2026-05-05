import logging

from pydtnn.layers.abstract.pool_2d_layer import AbstractPool2DLayer
from pydtnn.utils.constants import Array

__all__ = ("AveragePool2D",)

logger = logging.getLogger(__name__)


class AveragePool2D[T: Array](AbstractPool2DLayer[T]):
    pass
