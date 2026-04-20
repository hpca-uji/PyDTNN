from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.activations.tanh import Tanh
from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class TanhNumpy(Tanh[np.ndarray], ActivationNumpy):

    def _model_init(self, prev_shape, x=None):
        super()._model_init(prev_shape, x)
        # NOTE: This attribute only stores data, its value before the operation doesn't matters; it's initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros((self.model.batch_size, *prev_shape), dtype=self.model.dtype)

        self.memory_used += self._y.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = self._y[:x.shape[0], :]
        np.tanh(x, out=self.y,
                casting="unsafe", dtype=self.model.dtype)
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # return 1 - np.tanh(dy) ** 2
        np.tanh(dy, out=dy,
                casting="unsafe", dtype=dy.dtype)
        np.power(dy, 2, out=dy,
                 dtype=self.model.dtype)
        np.subtract(1, dy, out=dy,
                    dtype=self.model.dtype)

        return dy
