"""
CuPy implementation of the Rectified Linear Unit (ReLU) activation function.
"""

import logging

from pydtnn.backends.cupy.activations.abstract.activation import ActivationCupy
from pydtnn.backends.numpy.activations.relu import ReluNumpy
from pydtnn.libs import numpy as np

__all__ = ("ReluCupy",)

logger = logging.getLogger(__name__)


class ReluCupy(ReluNumpy, ActivationCupy):
    """
    ReLU activation layer implemented for CuPy backends.
    """

    def _model_init(self, prev_shape, x=None):
        """
        Initialize the layer model parameters.
        """
        super()._model_init(prev_shape, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass using CUDA kernels.
        """
        self.y = np.ascontiguousarray(self._y[: x.shape[0], :], dtype=self.model.dtype)
        self.mask = np.ascontiguousarray(self._mask[: x.shape[0], :], dtype=self.model.dtype)

        self.fwd_kernel(
            self.model.cuda_grid, self.model.cuda_block, (x, self.y, self.mask, self.y.size)
        )
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass using CUDA kernels.
        """
        self.bwd_kernel(self.model.cuda_grid, self.model.cuda_block, (dy, dy, self.mask, dy.size))
        return dy
