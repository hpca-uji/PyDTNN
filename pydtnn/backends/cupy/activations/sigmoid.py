import cupy as np

from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.backends.cupy.activations.activation import ActivationCUPY


class SigmoidCUPY(Sigmoid[np.ndarray], ActivationCUPY):

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        self.y: np.ndarray = None  # type: ignore (the value will be set in forward)

        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros(shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype)
        self.dx = np.zeros(shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:

        self.y = self._y[:x.shape[0], :]
        # y =  (1 / ( 1 + exp(-1*x)))

        np.multiply(x, -1, out=self.y)
        np.exp(self.y, out=self.y, dtype=self.model.dtype)
        np.add(self.y, 1, out=self.y)
        np.reciprocal(self.y, out=self.y, dtype=self.model.dtype)

        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:

        dx = self.dx[:dy.shape[0], :]

        # dx = dy * (y * (1 - y))
        np.subtract(1, self.y, out=dx)
        np.multiply(self.y, dx, out=dx, dtype=self.model.dtype)
        np.multiply(dy, dx, out=dx, dtype=self.model.dtype)

        return dx
