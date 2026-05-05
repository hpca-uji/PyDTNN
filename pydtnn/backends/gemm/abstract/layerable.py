from pydtnn.backends.gemm.abstract.base import BaseGemm
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy

__all__ = ("LayerableGemm",)


class LayerableGemm(LayerableNumpy, BaseGemm):
    ...
