import logging

from pydtnn.backends.gemm.abstract.layerable import LayerableGemm
from pydtnn.backends.numpy.layers.layer import LayerNumpy

__all__ = (
    "LayerGemm",
)

logger = logging.getLogger(__name__)


class LayerGemm(LayerNumpy, LayerableGemm):
    ...
