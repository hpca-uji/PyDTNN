"""Numpy backend implementation of the Log activation function."""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.libs import numpy as np
from pydtnn.activations.log import Log
from pydtnn.backends.numpy.activations.abstract.activation import ActivationNumpy
from pydtnn.utils.constants import ArrayShape

__all__ = ("LogNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class LogNumpy(Log[np.ndarray], ActivationNumpy):
    """Numpy-based Log activation layer."""

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initialize model parameters and allocate memory."""
        super()._model_init(prev_shape, x)

        self.y: np.ndarray
        self._y = np.zeros(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype)
        self.memory_used += self._y.nbytes

        self.exp_y: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.exp_y_shape = (self.model.batch_size, *self.shape)
        self.tmp_memory_used += (
            int(math.prod(self.exp_y_shape))
            * self.model.dtype.itemsize
        )

        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        """Allocate memory buffers after model initialization."""
        super()._post_init()

        with self.model.memory:
            self.exp_y = self.model.memory.ndarray(self.exp_y_shape, dtype=self.model.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass of the Log activation."""
        self.y = self._y[: x.shape[0], :]
        np.log(x, out=self.y)
        self.y = np.asarray(self.y, dtype=self.model.dtype, order="C")
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Perform the backward pass of the Log activation."""
        exp_y = self.exp_y[: dy.shape[0], :]
        np.negative(self.y, out=exp_y)
        np.exp(exp_y, out=exp_y, dtype=self.model.dtype)
        np.multiply(dy, exp_y, out=dy, dtype=self.model.dtype)
        return np.asarray(dy, dtype=self.model.dtype, order="C")
