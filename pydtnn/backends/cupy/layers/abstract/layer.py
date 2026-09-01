"""CuPy backend implementation for neural network layers."""

import logging

from pydtnn.backends.cupy.abstract.layerable import LayerableCupy
from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy

__all__ = ("LayerCupy",)

logger = logging.getLogger(__name__)


class LayerCupy(LayerNumpy, LayerableCupy):
    """Base class for layers using the CuPy backend."""
