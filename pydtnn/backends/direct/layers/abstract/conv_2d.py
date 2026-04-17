import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.backends.direct.layers.layer import LayerDirect


class AbstractConv2DDirect(AbstractConv2DNumpy, LayerDirect):
    ...
