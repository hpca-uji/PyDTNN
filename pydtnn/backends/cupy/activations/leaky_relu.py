"""CuPy implementation of the Leaky ReLU activation function."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.cupy.activations.abstract.activation import ActivationCupy
from pydtnn.backends.numpy.activations.leaky_relu import LeakyReluNumpy
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape

__all__ = ("LeakyReluCupy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class LeakyReluCupy(LeakyReluNumpy, ActivationCupy):
    """Leaky ReLU activation layer implemented for CuPy backends."""

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initialize the layer parameters and buffers."""
        super()._model_init(prev_shape, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass using a CUDA kernel."""
        self.y = np.ascontiguousarray(self._y[: x.shape[0], :], dtype=self.model.dtype)
        self.mask = np.ascontiguousarray(self._mask[: x.shape[0], :], dtype=self.model.dtype)

        self.fwd_kernel(
            self.model.cuda_grid,
            self.model.cuda_block,
            (x, self.y, self.mask, self.negative_slope, x.size),
        )
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Perform the backward pass using a CUDA kernel."""
        self.bwd_kernel(
            self.model.cuda_grid,
            self.model.cuda_block,
            (dy, dy, self.mask, self.negative_slope, dy.size),
        )
        return dy
