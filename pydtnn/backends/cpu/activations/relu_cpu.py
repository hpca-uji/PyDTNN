from pydtnn.activations.relu import Relu
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from pydtnn.cython_modules import relu_cython
import numpy as np

class ReluCPU(ActivationCPU, Relu):

    def __init__(self, shape:tuple[int, ...]=(1,)):
        super().__init__(shape)
        self.mask:np.ndarray = None

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self._y = np.empty((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)
        self._mask = np.empty((self.model.batch_size, *self.prev_shape), dtype=np.int8)
        self.dx = np.empty((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)

    def forward(self, x:np.ndarray) -> np.ndarray:
        n = x.shape[0]
        self.y = self._y[:n, :]
        self.mask = self._mask[:n, :]
        relu_cython(x.reshape(-1, copy=False), self.y.reshape(-1, copy=False), self.mask.reshape(-1, copy=False))
        return self.y

    def backward(self, dy:np.ndarray) -> np.ndarray:
        dx = self.dx[:dy.shape[0], :]
        np.multiply(dy, self.mask, out=dx)
        return dx
