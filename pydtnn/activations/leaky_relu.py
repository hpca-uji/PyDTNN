"""Leaky ReLU activation layer implementation."""

import logging

from pydtnn.activations.relu import Relu
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("LeakyRelu",)

logger = logging.getLogger(__name__)


class LeakyRelu[T: Array](Relu[T]):  # noqa: D101 (generics not detected)
    """Leaky Rectified Linear Unit activation layer."""

    def __init__(self, shape: ArrayShape = (1,), negative_slope: float = 0.01) -> None:
        """Initializes the LeakyRelu layer.

        Args:
            shape: The shape of the input tensor.
            negative_slope: The slope for negative input values.
        """
        super().__init__(shape)
        self.negative_slope: float = negative_slope
