"""
Winograd backend implementation for neural network layers.
"""
import logging

from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.backends.winograd.abstract.layerable import LayerableWinograd

__all__ = ("LayerWinograd",)

logger = logging.getLogger(__name__)


class LayerWinograd(LayerNumpy, LayerableWinograd):
    """
    Base class for layers utilizing Winograd convolution algorithms.
    """
    ...