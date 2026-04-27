from pydtnn.backends.gemm.abstract.base import BaseGemm
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy


class LayerableGemm(LayerableNumpy, BaseGemm):
    ...
