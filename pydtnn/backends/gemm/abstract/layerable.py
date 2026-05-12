"""Abstract base class for GEMM-based layerable components in PyDTNN."""

from pydtnn.backends.gemm.abstract.base import BaseGemm
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy

"""Abstract base class for GEMM-based layerable components in PyDTNN."""

__all__ = ("LayerableGemm",)


class LayerableGemm(LayerableNumpy, BaseGemm):
    """
    A mixin or base class that combines NumPy layerable functionality 
    with General Matrix Multiply (GEMM) abstract interfaces.
    """
    ...