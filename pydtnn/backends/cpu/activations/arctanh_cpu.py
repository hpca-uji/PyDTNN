import numpy as np

from pydtnn.activations.arctanh import Arctanh
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from numpy import ndarray


class ArctanhCPU(ActivationCPU, Arctanh):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self._y = np.empty(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype)

    def forward(self, x: ndarray) -> ndarray:
        self.y = self._y[:x.shape[0], :]
        np.arctan(x, out=self.y, casting="unsafe", dtype=x.dtype)
        return self.y

    def backward(self, dy: ndarray) -> ndarray:
        # return 1 / (1 + dy ** 2)
        dy **= 2
        dy += 1
        np.reciprocal(dy, out=dy, casting="unsafe", dtype=self.model.dtype)
        return dy
