"""
Softmax activation module for PyDTNN.
"""

import logging

from pydtnn.activations.abstract.activation import Activation
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("Softmax",)

logger = logging.getLogger(__name__)


class Softmax[T: Array](Activation[T]):
    """
    Softmax activation layer that computes the normalized exponential of the input.
    """

    def __init__(self, shape: ArrayShape = (1,), axis: int = 1):
        """
        Initializes the Softmax layer.

        Args:
            shape: The expected shape of the input tensor.
            axis: The axis along which the softmax computation is performed.
        """
        super().__init__(shape)
        self.axis_dim = axis
