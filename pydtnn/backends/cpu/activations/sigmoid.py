from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.backends.cpu.activations.activation import ActivationCPU


class SigmoidCPU(Sigmoid[np.ndarray], ActivationCPU):

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)

        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y: np.ndarray = np.zeros(shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype)
        self.real_memory_size += self._y.nbytes

        self.dx: np.ndarray = np.zeros(shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype)
        self.real_memory_size += self.dx.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y: np.ndarray = self._y[:x.shape[0], :]
        # y = (1 / ( 1 + exp(-1*x)))
        np.multiply(-1, x, out=self.y)
        np.exp(self.y, out=self.y)
        np.add(1, self.y, out=self.y)
        np.reciprocal(self.y, out=self.y)
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx: np.ndarray = self.dx[:dy.shape[0], :]
        # dx = dy * (y * (1 - y))
        np.subtract(1, self.y, out=dx)
        np.multiply(self.y, dx, out=dx)
        np.multiply(dy, dx, out=dx)
        return dx
