"""
Cython-based implementation of neural network layers for PyDTNN.
"""

import logging

from pydtnn.backends.cython.abstract.layerable import LayerableCython
from pydtnn.backends.numpy.layers.layer import LayerNumpy

__all__ = ("LayerCython",)

logger = logging.getLogger(__name__)


class LayerCython(LayerNumpy, LayerableCython):
    """
    Base class for Cython-accelerated neural network layers.
    """

    ...
