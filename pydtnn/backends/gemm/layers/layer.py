from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.backends.gemm.abstract.layerable import LayerableGemm
import logging
logger = logging.getLogger(__name__)


class LayerGemm(LayerNumpy, LayerableGemm):
    ...
