import logging

from pydtnn.backends.cython.layers.abstract.conv_2d import AbstractConv2DCython
from pydtnn.backends.numpy.layers.conv_2d import Conv2DNumpy

__all__ = (
    "Conv2DCython",
)

logger = logging.getLogger(__name__)


class Conv2DCython(Conv2DNumpy, AbstractConv2DCython):
    ...
