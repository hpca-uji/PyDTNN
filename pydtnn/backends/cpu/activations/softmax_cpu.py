import numpy as np

from pydtnn.activations.softmax import Softmax
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from pydtnn.utils.types import ArrayShape

class SoftmaxCPU(ActivationCPU, Softmax[np.ndarray]):

    def __init__(self, shape: ArrayShape = (1,)):
        super().__init__(shape)

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        self.y: np.ndarray

        self.axis_dim = 1
        shape_intermediate_ops = list(self.shape)
        shape_intermediate_ops[self.axis_dim - 1] = 1

        # NOTE: These attributes only store data, their value before the operation doesn't matter.
        self._y = np.zeros(shape=(self.model.batch_size, *self.shape),
                           dtype=self.model.dtype, order="C")
        self.mul_dy = np.zeros(shape=(self.model.batch_size, *self.shape),
                               dtype=self.model.dtype, order="C")
        self.max_x = np.zeros(shape=(self.model.batch_size, *shape_intermediate_ops),
                              dtype=self.model.dtype, order="C")
        self.sum_y = np.zeros(shape=(self.model.batch_size, *shape_intermediate_ops),
                              dtype=self.model.dtype, order="C")
        self.sum_dy = np.zeros(shape=(self.model.batch_size, *shape_intermediate_ops),
                               dtype=self.model.dtype, order="C")

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
                  dtype=self.model.dtype, order="C")
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
                    dtype=self.model.dtype, order="C")

        return dy
