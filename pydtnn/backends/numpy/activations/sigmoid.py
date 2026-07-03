"""NumPy backend implementation of the Sigmoid activation function."""

import logging
from typing import TYPE_CHECKING

from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.backends.numpy.activations.abstract.activation import ActivationNumpy
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape

__all__ = ("SigmoidNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class SigmoidNumpy(Sigmoid[np.ndarray], ActivationNumpy):
    """NumPy-based Sigmoid activation layer."""

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initialize layer buffers and memory tracking."""
        super()._model_init(prev_shape, x)

        # NOTE: These attributes only store data, their values before the
        # operation doesn't matter; they're initalized due avoid warnings in
        # "LayerAndActivationBase.export".
        self._y: np.ndarray = np.zeros(
            shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype
        )
        self.memory_used += self._y.nbytes

        self.dx: np.ndarray = np.zeros(
            shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype
        )
        self.memory_used += self.dx.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward pass of the sigmoid function."""
        self.y: np.ndarray = self._y[: x.shape[0], :]
        # y = (1 / ( 1 + exp(-1*x)))
        np.multiply(-1, x, out=self.y)
        np.exp(self.y, out=self.y)
        np.add(1, self.y, out=self.y)
        np.reciprocal(self.y, out=self.y)
        self.y = np.asarray(self.y, dtype=self.model.dtype, order="C")
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Compute the backward pass of the sigmoid function."""
        dx: np.ndarray = self.dx[: dy.shape[0], :]
        # dx = dy * (y * (1 - y))
        np.subtract(1, self.y, out=dx)
        np.multiply(self.y, dx, out=dx)
        np.multiply(dy, dx, out=dx)

        return np.asarray(dx, dtype=self.model.dtype, order="C")
