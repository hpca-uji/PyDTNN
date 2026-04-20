import logging

from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.backends.winograd.layers.layer import LayerWinograd
logger = logging.getLogger(__name__)


class AbstractConv2DWinograd(AbstractConv2DNumpy, LayerWinograd):
    ...
