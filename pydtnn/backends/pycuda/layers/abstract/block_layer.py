"""
PyCUDA implementation of abstract block layers for the PyDTNN framework.
"""

import logging

from pydtnn.backends.pycuda.layers.layer import LayerPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer

__all__ = ("AbstractBlockLayerPycuda",)

logger = logging.getLogger(__name__)


class AbstractBlockLayerPycuda(AbstractBlockLayer[TensorArray], LayerPycuda):
    """
    Base class for PyCUDA-accelerated block layers.
    """

    pass
