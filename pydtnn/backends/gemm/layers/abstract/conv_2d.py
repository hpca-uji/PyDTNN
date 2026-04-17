import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.backends.gemm.layers.layer import LayerGemm


class AbstractConv2DGemm(AbstractConv2DNumpy, LayerGemm):
    ...
