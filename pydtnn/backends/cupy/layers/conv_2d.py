from pydtnn.utils.constants import DTYPE2CTYPE
from pydtnn.utils.constants import ArrayShape
from pydtnn.backends.numpy.layers.conv_2d import Conv2DNumpy
from pydtnn.backends.cupy.layers.layer import LayerCupy
import cupy as np
from cupy.cuda import Stream  # type: ignore
import logging

from pydtnn.backends.cupy.layers.abstract.conv_2d import AbstractConv2DCupy
logger = logging.getLogger(__name__)


class Conv2DCupy(Conv2DNumpy, AbstractConv2DCupy, LayerCupy):
    ...