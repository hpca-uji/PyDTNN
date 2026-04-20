from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.activations.sigmoid import Sigmoid
from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class SigmoidNumpy(Sigmoid[np.ndarray], ActivationNumpy):

    def _model_init(self, prev_shape, x=None):
        super()._model_init(prev_shape, x)

        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y: np.ndarray = np.zeros(shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype)
        self.memory_used += self._y.nbytes

        self.dx: np.ndarray = np.zeros(shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype)
        self.memory_used += self.dx.nbytes

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
