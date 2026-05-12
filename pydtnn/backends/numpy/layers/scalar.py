"""
Numpy backend implementation for scalar multiplication layers.
"""
import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.layers.scalar import Scalar
from pydtnn.libs import numpy as np

__all__ = ("ScalarNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class ScalarNumpy(Scalar[np.ndarray], LayerNumpy):
    """
    Numpy-based scalar multiplication layer.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the ScalarNumpy layer.
        """
        super().__init__(*args, **kwargs)

    def _model_init(self, prev_shape, x):
        """
        Initialize model performance metrics.

        Args:
            prev_shape: Shape of the previous layer output.
            x: Input data.
        """
        super()._model_init(prev_shape, x)
        # Performance model
        self.fwd_time: np.ndarray = None  # type: ignore # Not yet
        self.bwd_time: np.ndarray = None  # type: ignore # Not yet

    def forward(self, x):
        """
        Perform forward pass by scaling the input.

        Args:
            x: Input array.

        Returns:
            Scaled input array.
        """
        return x * self.scale

    def backward(self, dy):
        """
        Perform backward pass by scaling the gradient.

        Args:
            dy: Gradient of the loss with respect to the output.

        Returns:
            Scaled gradient.
        """
        return dy * self.scale