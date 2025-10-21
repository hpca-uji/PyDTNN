from pydtnn.activations.relu import Relu
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from pydtnn.cython import relu_cython
import numpy as np
from pydtnn.utils.types import shape_t

class ReluCPU(ActivationCPU, Relu):

    def __init__(self, shape: shape_t = (1,)):
        super().__init__(shape)
        self.mask: np.ndarray = None

    def initialize(self, prev_shape, x = None):
        super().initialize(prev_shape, x)
        self._y = np.empty((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")
        self._mask = np.empty((self.model.batch_size, *self.prev_shape), dtype=np.int8, order="C")
        self.dx = np.empty((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")

    def forward(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        self.y = self._y[:n, :]
        self.mask = self._mask[:n, :]
        relu_cython(x.reshape(-1, copy=False, order="C"), self.y.reshape(-1, copy=False, order="C"), self.mask.reshape(-1, copy=False, order="C"))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx = self.dx[:dy.shape[0], :]
        np.multiply(dy, self.mask, out=dx, dtype=self.model.dtype, order="C")
        return dx
