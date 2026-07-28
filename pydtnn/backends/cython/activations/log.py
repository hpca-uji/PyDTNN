"""Cython implementation of the Logarithmic activation function."""

import logging

from pydtnn.backends.cython.activations.abstract.activation import ActivationCython
from pydtnn.backends.cython.utils.log_sigmoid_activation_cython import log_sigmoid_bwd_cython, log_sigmoid_fwd_cython
from pydtnn.backends.numpy.activations.log_sigmoid import LogSigmoidNumpy
from pydtnn.libs import numpy as np

__all__ = ("LogSigmoidCython",)

logger = logging.getLogger(__name__)


class LogSigmoidCython(LogSigmoidNumpy, ActivationCython):
    """Cython-accelerated Logarithmic activation layer."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass using Cython optimization.

        Args:
            x: Input array.

        Returns:
            The natural logarithm of the input.
        """
        self.y: np.ndarray = self._y[: x.shape[0], :]
        log_sigmoid_fwd_cython(x.reshape(-1, copy=False), self.y.reshape(-1, copy=False))
        return np.asarray(self.y, dtype=self.model.dtype, order="C")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass using Cython optimization.

        Args:
            dy: Gradient of the loss with respect to the output.

        Returns:
            Gradient of the loss with respect to the input.
        """
        dx: np.ndarray = self.dx[: dy.shape[0], :]
        log_sigmoid_bwd_cython(dy.reshape(-1, copy=False), dx.reshape(-1, copy=False))
        return np.asarray(dx, dtype=self.model.dtype, order="C")
