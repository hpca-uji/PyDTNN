import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.cython.layers.abstract.conv_2d import AbstractConv2DCython
from pydtnn.backends.numpy.layers.conv_2d import Conv2DNumpy


class Conv2DCython(Conv2DNumpy, AbstractConv2DCython):
    ...
