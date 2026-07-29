"""Numpy backend implementation of the Log activation function."""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.activations.log_softmax import LogSoftmax
from pydtnn.backends.numpy.activations.abstract.activation import ActivationNumpy
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape

__all__ = ("LogSoftmaxNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class LogSoftmaxNumpy(LogSoftmax[np.ndarray], ActivationNumpy):
    """Numpy-based Log Softmax activation layer."""

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initialize model parameters and allocate memory for temporary buffers."""
        super()._model_init(prev_shape, x)
        self.y: np.ndarray

        shape_intermediate_ops = list(self.shape)
        shape_intermediate_ops[self.axis_dim - 1] = 1

        # NOTE: These attributes only store data, their value before the operation
        # doesn't matter; they're initalized due avoid warnings in
        # "LayerAndActivationBase.export".
        self._y = np.zeros(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype)
        self.memory_used += self._y.nbytes

        # Temp_variables
        self.max_x: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.sum_y: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.exp_y: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.sum_dy: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]

        self.temp_shape = (self.model.batch_size, *shape_intermediate_ops)
        sum_y_shape = max_x_shape = self.temp_shape
        self.tmp_memory_used += (
            int(math.prod(max_x_shape) + math.prod(sum_y_shape)) * self.model.dtype.itemsize
        )

        self.exp_y_shape = (self.model.batch_size, *self.shape)
        self.sum_dy_shape = (self.model.batch_size, 1)
        self.tmp_memory_used += (
            int(math.prod(self.exp_y_shape) + math.prod(self.sum_dy_shape))
            * self.model.dtype.itemsize
        )

        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        """Allocate memory buffers after model initialization."""
        super()._post_init()
        with self.model.memory:
            self.max_x = self.model.memory.ndarray(self.temp_shape, dtype=self.model.dtype)
            self.sum_y = self.model.memory.ndarray(self.temp_shape, dtype=self.model.dtype)
            self.exp_y = self.model.memory.ndarray(self.exp_y_shape, dtype=self.model.dtype)
            self.sum_dy = self.model.memory.ndarray(self.sum_dy_shape, dtype=self.model.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass of the LogSoftmax activation."""
        #  self.y = log(np.exp(x - np.max(x, axis=1, keepdims=True)) /
        #               np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)) =
        #         = log(np.exp(x - np.max(x, axis=1, keepdims=True))) - 
        #           log(np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)) =
        #         = x - np.max(x, axis=1, keepdims=True) -
        #           log(np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True))
        #  return self.y

        self.y = self._y[: x.shape[0], :]
        max_x = self.max_x[: x.shape[0], :]
        sum_y = self.sum_y[: x.shape[0], :]

        #  x = x - np.max(x, axis=1, keepdims=True)
        np.max(x, axis=self.axis_dim, keepdims=True, out=max_x)
        np.subtract(x, max_x, out=x, dtype=self.model.dtype)

        #  sum_y = log(np.sum(np.exp(x), axis=1, keepdims=True))
        np.exp(x, out=self.y, dtype=self.model.dtype)
        np.sum(self.y, axis=self.axis_dim, keepdims=True, out=sum_y)
        np.log(sum_y, out=sum_y)

        #  self.y = x - sum_y
        np.subtract(x, sum_y, out=self.y, dtype=self.model.dtype)
        self.y = np.asarray(self.y, dtype=self.model.dtype, order="C")
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Perform the backward pass of the Log Softmax activation."""

        sum_dy = self.sum_dy[: dy.shape[0], :]
        exp_y = self.exp_y[: dy.shape[0], :]

        np.exp(self.y, dtype=self.model.dtype, out = exp_y)

        np.sum(dy, axis=self.axis_dim, keepdims=True, out=sum_dy)
        np.multiply(exp_y, sum_dy, out=exp_y)

        np.subtract(dy, exp_y, out=dy)

        return np.asarray(dy, dtype=self.model.dtype, order="C")
