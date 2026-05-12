"""
Cython-accelerated ReLU activation implementation.
"""
import logging

from pydtnn.backends.cython.activations.activation import ActivationCython
from pydtnn.backends.cython.utils.relu_cython import relu_cython
from pydtnn.backends.numpy.activations.relu import ReluNumpy
from pydtnn.libs import numpy as np

__all__ = ("ReluCython",)

logger = logging.getLogger(__name__)


class ReluCython(ReluNumpy, ActivationCython):
    """
    ReLU activation layer using Cython for optimized computation.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the ReLU activation.

        Args:
            x: Input tensor.

        Returns:
            The result of the ReLU activation applied to the input.
        """
        n = x.shape[0]
        self.y = self._y[:n, :]
        self.mask = self._mask[:n, :]
        mask: np.ndarray[tuple[int], np.int8] = self.mask.reshape(-1, copy=False)
        relu_cython(x.reshape(-1, copy=False), self.y.reshape(-1, copy=False), mask)
        return self.y