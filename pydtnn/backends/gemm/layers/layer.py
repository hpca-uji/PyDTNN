import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.gemm.abstract.layerable import LayerableGemm
from pydtnn.backends.numpy.layers.layer import LayerNumpy


class LayerGemm(LayerNumpy, LayerableGemm):
    ...
