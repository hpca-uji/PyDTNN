import numpy as np

from pydtnn.activations.tanh import Tanh
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU


class TanhCPU(ActivationCPU, Tanh):

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self._y = np.empty((self.model.batch_size, *prev_shape), dtype=self.model.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = self._y[:x.shape[0], :]
        np.tanh(x, out=self.y, casting="unsafe", dtype=self.model.dtype)
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # return 1 - np.tanh(dy) ** 2
        np.tanh(dy, out=dy, casting="unsafe", dtype=dy.dtype)
        np.power(dy, 2, out=dy)
        np.subtract(1, dy, out=dy)

        return dy
