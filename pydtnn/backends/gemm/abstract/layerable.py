from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy
from pydtnn.backends.gemm.abstract.base import BaseGemm


class LayerableGemm(LayerableNumpy, BaseGemm):
    ...
