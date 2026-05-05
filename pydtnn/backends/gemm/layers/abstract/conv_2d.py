import logging

from pydtnn.backends.gemm.layers.layer import LayerGemm
from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy

__all__ = ("AbstractConv2DGemm",)

logger = logging.getLogger(__name__)


class AbstractConv2DGemm(AbstractConv2DNumpy, LayerGemm): ...
