"""
NumPy backend implementation of the Leaky ReLU activation function.
"""
import logging
from typing import TYPE_CHECKING

from pydtnn.activations.leaky_relu import LeakyRelu
from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.libs import numpy as np

__all__ = ("LeakyReluNumpy",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class LeakyReluNumpy(LeakyRelu[np.ndarray], ActivationNumpy):
    """
    NumPy-based Leaky ReLU activation layer.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the LeakyReluNumpy layer.
        """
        super().__init__(*args, **kwargs)

    def _model_init(self, prev_shape, x=None):
        """
        Initialize internal buffers for the forward and backward passes.
        """
        super()._model_init(prev_shape, x)
        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)
        self._mask = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)

        self.memory_used += self._y.nbytes + self._mask.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the Leaky ReLU activation.

        Args:
            x: Input tensor.

        Returns:
            The activated output tensor.
        """
        self.y = self._y[: x.shape[0], :]
        self.mask = self._mask[: x.shape[0], :]

        negatives = x < 0

        self.y[~negatives] = x
        self.y[negatives] = x * self.negative_slope

        np.greater(x, 0, out=self.mask, dtype=np.int8)
        self.mask[negatives] = self.negative_slope

        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass (gradient) of the Leaky ReLU activation.

        Args:
            dy: Gradient of the loss with respect to the output.

        Returns:
            Gradient of the loss with respect to the input.
        """
        # return dy * self.mask
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype)
        return dy