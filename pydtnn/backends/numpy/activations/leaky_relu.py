from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.activations.leaky_relu import LeakyRelu
import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class LeakyReluNumpy(LeakyRelu[np.ndarray], ActivationNumpy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _model_init(self, prev_shape, x=None):
        super()._model_init(prev_shape, x)
        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)
        self._mask = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)

        self.memory_used += self._y.nbytes + self._mask.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = self._y[:x.shape[0], :]
        self.mask = self._mask[:x.shape[0], :]

        negatives = (x < 0)

        self.y[~negatives] = x
        self.y[negatives] = x * self.negative_slope

        np.greater(x, 0, out=self.mask, dtype=np.int8)
        self.mask[negatives] = self.negative_slope

        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # return dy * self.mask
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype)
        return dy
