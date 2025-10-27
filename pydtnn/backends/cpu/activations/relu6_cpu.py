from pydtnn.activations.relu6 import Relu6
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from pydtnn.cython import capped_relu_cython
import numpy as np


class Relu6CPU(Relu6, ActivationCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        self._y = np.empty((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")
        self._mask = np.empty((self.model.batch_size, *self.prev_shape), dtype=np.int8, order="C")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y: np.ndarray = self._y[:x.shape[0], :]
        self.mask: np.ndarray = self._mask[:x.shape[0], :]
        capped_relu_cython(x.reshape(-1, copy=False, order="C"),
                           self.y.reshape(-1, copy=False, order="C"),
                           self.mask.reshape(-1, copy=False, order="C"),
                           self.cap)
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # return dy * self.mask
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype, order="C")
        return dy
