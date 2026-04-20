import math
from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.activations.softmax import Softmax
from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class SoftmaxNumpy(Softmax[np.ndarray], ActivationNumpy):
    def _model_init(self, prev_shape, x=None):
        super()._model_init(prev_shape, x)
        self.y: np.ndarray

        shape_intermediate_ops = list(self.shape)
        shape_intermediate_ops[self.axis_dim - 1] = 1

        # NOTE: These attributes only store data, their value before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype)
        self.memory_used += self._y.nbytes

        # Temp_variables
        self.max_x: np.ndarray = None  # type: ignore (they will be intialized later)
        self.sum_y: np.ndarray = None  # type: ignore (they will be intialized later)
        self.mul_dy: np.ndarray = None  # type: ignore (they will be intialized later)
        self.sum_dy: np.ndarray = None  # type: ignore (they will be intialized later)

        self.temp_shape = (self.model.batch_size, *shape_intermediate_ops)
        sum_y_shape = max_x_shape = self.temp_shape
        self.tmp_memory_used += int(math.prod(max_x_shape) + math.prod(sum_y_shape)) * self.model.dtype.itemsize

        self.mul_dy_shape = (self.model.batch_size, *self.shape)
        self.sum_dy_shape = (self.model.batch_size, *shape_intermediate_ops)
        self.tmp_memory_used += int(math.prod(self.mul_dy_shape) + math.prod(self.sum_dy_shape)) * self.model.dtype.itemsize

        self.memory_used += self.tmp_memory_used

    def _post_init(self):
        super()._post_init()
        with self.model.memory:
            self.max_x = self.model.memory.ndarray(self.temp_shape, dtype=self.model.dtype)
            self.sum_y = self.model.memory.ndarray(self.temp_shape, dtype=self.model.dtype)
            self.mul_dy = self.model.memory.ndarray(self.mul_dy_shape, dtype=self.model.dtype)
            self.sum_dy = self.model.memory.ndarray(self.sum_dy_shape, dtype=self.model.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # self.y = np.exp(x - np.max(x, axis=1, keepdims=True))
        # self.y /= np.sum(self.y, axis=1, keepdims=True)
        # return self.y
        self.y = self._y[:x.shape[0], :]
        max_x = self.max_x[:x.shape[0], :]
        sum_y = self.sum_y[:x.shape[0], :]

        np.max(x, axis=self.axis_dim, keepdims=True, out=max_x)
        np.subtract(x, max_x, out=x,
                    dtype=self.model.dtype)
        np.exp(x, out=self.y,
               dtype=self.model.dtype)
        np.sum(self.y, axis=self.axis_dim, keepdims=True, out=sum_y)
        np.divide(self.y, sum_y, out=self.y,
                  dtype=self.model.dtype)
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # return self.y * (dy - (dy * self.y).sum(axis=1, keepdims=True))
        sum_dy = self.sum_dy[:dy.shape[0], :]
        mul_dy = self.mul_dy[:dy.shape[0], :]

        np.multiply(dy, self.y, out=mul_dy,
                    dtype=self.model.dtype)
        mul_dy.sum(axis=self.axis_dim, keepdims=True, out=sum_dy)
        np.subtract(dy, sum_dy, out=dy,
                    dtype=self.model.dtype)
        np.multiply(self.y, dy, out=dy,
                    dtype=self.model.dtype)

        return dy
