import logging
logger = logging.getLogger(__name__)

from pydtnn.layers.abstract.conv_2d import AbstractConv2D
from pydtnn.utils.constants import Array


class Conv2DPointwise[T: Array](AbstractConv2D[T]):
    pass
