from pydtnn.utils.constants import Array
from pydtnn.layers.abstract.conv_2d import AbstractConv2D
import logging
logger = logging.getLogger(__name__)


class Conv2DDepthwise[T: Array](AbstractConv2D[T]):
    pass
