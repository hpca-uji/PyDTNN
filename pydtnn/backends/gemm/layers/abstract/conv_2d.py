"""Abstract base class for 2D convolutional layers using GEMM-based operations."""

import logging

from pydtnn.backends.gemm.layers.abstract.layer import LayerGemm
from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy

"""Abstract base class for 2D convolutional layers using GEMM-based operations."""

__all__ = ("AbstractConv2DGemm",)

logger = logging.getLogger(__name__)


class AbstractConv2DGemm(AbstractConv2DNumpy, LayerGemm):
    """Abstract base class for 2D convolutional layers implemented via General Matrix Multiply (GEMM) operations."""
