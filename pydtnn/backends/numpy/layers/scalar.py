"""Numpy backend implementation for scalar multiplication layers."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy
from pydtnn.layers.scalar import Scalar
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape

__all__ = ("ScalarNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class ScalarNumpy(Scalar[np.ndarray], LayerNumpy):
    """Numpy-based scalar multiplication layer."""

    def __init__(self, shape: ArrayShape = (1,), scale: float = 1.0) -> None:
        """Initialize the ScalarNumpy layer."""
        super().__init__(shape, scale)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform forward pass by scaling the input.

        Args:
            x: Input array.

        Returns:
            Scaled input array.
        """
        np.multiply(x, self.scale, out=x)
        return np.asarray(x, dtype=self.model.dtype, order="C")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Perform backward pass by scaling the gradient.

        Args:
            dy: Gradient of the loss with respect to the output.

        Returns:
            Scaled gradient.
        """
        np.multiply(dy, self.scale, out=dy)
        return np.asarray(dy, dtype=self.model.dtype, order="C")
