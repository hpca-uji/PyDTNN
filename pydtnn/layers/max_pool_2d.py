import logging
logger = logging.getLogger(__name__)

from pydtnn.layers.abstract.pool_2d_layer import AbstractPool2DLayer
from pydtnn.utils.constants import Array


class MaxPool2D[T: Array](AbstractPool2DLayer[T]):
    pass
