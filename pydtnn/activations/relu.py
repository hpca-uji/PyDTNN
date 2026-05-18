"""Rectified Linear Unit (ReLU) activation function module."""

import logging

from pydtnn.activations.abstract.activation import Activation
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("Relu",)

logger = logging.getLogger(__name__)


class Relu[T: Array](Activation[T]):
    """Rectified Linear Unit activation layer."""

    def __init__(self, shape: ArrayShape = (1,)):
        """Initializes the ReLU layer with a given shape.

        Args:
            shape: The shape of the input tensor.
        """
        super().__init__(shape)
        # Will be initalized in "initialize"
        self.mask: T = None  # type: ignore
