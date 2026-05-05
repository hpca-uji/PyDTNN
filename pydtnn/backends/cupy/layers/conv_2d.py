import logging

from pydtnn.backends.cupy.layers.abstract.conv_2d import AbstractConv2DCupy
from pydtnn.backends.cupy.layers.layer import LayerCupy
from pydtnn.backends.numpy.layers.conv_2d import Conv2DNumpy

__all__ = ("Conv2DCupy",)

logger = logging.getLogger(__name__)


class Conv2DCupy(Conv2DNumpy, AbstractConv2DCupy, LayerCupy):
    ...
