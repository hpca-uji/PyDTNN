"""Cython implementation of the Sigmoid activation function."""

import logging

from pydtnn.backends.cython.activations.abstract.activation import ActivationCython
from pydtnn.backends.cython.utils.sigmoid_cython import sigmoid_bwd_cython, sigmoid_fwd_cython
from pydtnn.backends.numpy.activations.sigmoid import SigmoidNumpy
from pydtnn.libs import numpy as np

__all__ = ("SigmoidCython",)

logger = logging.getLogger(__name__)


class SigmoidCython(SigmoidNumpy, ActivationCython):
    """Cython-accelerated Sigmoid activation layer."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the Sigmoid activation.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying Sigmoid.
        """
        self.y: np.ndarray = self._y[: x.shape[0], :]
        sigmoid_fwd_cython(x.reshape(-1, copy=False), self.y.reshape(-1, copy=False))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of the Sigmoid activation.

        Args:
            dy: Gradient of the loss with respect to the output.

        Returns:
            Gradient of the loss with respect to the input.
        """
        dx: np.ndarray = self.dx[: dy.shape[0], :]
        sigmoid_bwd_cython(
            dy.reshape(-1, copy=False), self.y.reshape(-1, copy=False), dx.reshape(-1, copy=False)
        )
        return dx
