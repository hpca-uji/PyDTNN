"""GEMM-based layer implementation for the PyDTNN framework."""

import logging

from pydtnn.backends.gemm.abstract.layerable import LayerableGemm
from pydtnn.backends.numpy.layers.layer import LayerNumpy

"""GEMM-based layer implementation for the PyDTNN framework."""

__all__ = ("LayerGemm",)

logger = logging.getLogger(__name__)


class LayerGemm(LayerNumpy, LayerableGemm):
    """
    Base class for layers utilizing General Matrix Multiply (GEMM) operations.

    Inherits from LayerNumpy for standard array operations and LayerableGemm
    for GEMM-specific interface requirements.
    """

    ...
