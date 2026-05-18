"""
Cython-based activation layer implementations for the PyDTNN framework.
"""

import logging

from pydtnn.backends.cython.abstract.layerable import LayerableCython
from pydtnn.backends.numpy.activations.abstract.activation import ActivationNumpy

__all__ = ("ActivationCython",)

logger = logging.getLogger(__name__)


class ActivationCython(ActivationNumpy, LayerableCython):
    """
    Base class for Cython-accelerated activation layers.
    """

    ...
