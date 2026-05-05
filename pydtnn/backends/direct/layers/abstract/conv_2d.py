import logging

from pydtnn.backends.direct.layers.layer import LayerDirect
from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy

__all__ = ("AbstractConv2DDirect",)

logger = logging.getLogger(__name__)


class AbstractConv2DDirect(AbstractConv2DNumpy, LayerDirect):
    ...
