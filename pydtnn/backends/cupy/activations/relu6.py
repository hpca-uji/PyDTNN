"""CuPy implementation of the ReLU6 activation function."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.cupy.activations.abstract.activation import ActivationCupy
from pydtnn.backends.numpy.activations.relu6 import Relu6Numpy
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape

__all__ = ("Relu6Cupy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class Relu6Cupy(Relu6Numpy, ActivationCupy):
    """ReLU6 activation layer implemented using CuPy for GPU acceleration."""

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initialize the layer parameters and buffers."""
        super()._model_init(prev_shape, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass using a CUDA kernel."""
        self.y = np.ascontiguousarray(self._y[: x.shape[0], :], dtype=self.model.dtype)
        self.mask = np.ascontiguousarray(self._mask[: x.shape[0], :], dtype=self.model.dtype)

        self.fwd_kernel(
            self.model.cuda_grid, self.model.cuda_block, (x, self.y, self.mask, self.cap, x.size)
        )
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Perform the backward pass using a CUDA kernel."""
        self.bwd_kernel(self.model.cuda_grid, self.model.cuda_block, (dy, dy, self.mask, dy.size))
        return dy
