import logging
logger = logging.getLogger(__name__)

from pydtnn.layers.conv_2d import Conv2D
from pydtnn.utils.constants import Array


class Conv2DDepthwise[T: Array](Conv2D[T]):
    pass
