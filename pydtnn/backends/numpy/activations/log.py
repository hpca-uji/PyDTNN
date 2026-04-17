from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.activations.log import Log
from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class LogNumpy(Log[np.ndarray], ActivationNumpy):

    def _model_init(self, prev_shape, x=None):
        super()._model_init(prev_shape, x)
        # NOTE: These attributes only store data, their value before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self.y = np.zeros(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype)
        self.memory_used += self.y.nbytes

        if not self.model.evaluate_only:
            self.dx = np.zeros(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype)
            self.memory_used += self.dx.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        # def forward(self, x: np.ndarray) -> np.ndarray:
        y = self.y[:x.shape[0], :]
        # y = np.log(1 / (1 + np.exp(-x)))
        np.multiply(x, -1, out=x,
                    dtype=self.model.dtype)
        np.exp(x, out=x,
               dtype=self.model.dtype)
        np.add(x, 1, out=x,
               dtype=self.model.dtype)
        np.log(x, out=y,
               dtype=self.model.dtype)
        # NOTE: Log propierty: "log(a / b) = log(a) - log(b)", and "log(1) = 0 ==>
        #                       ==> "log(a / b) = - log(b)""
        np.multiply(y, -1, out=y,
                    dtype=self.model.dtype)
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # return 1 / (np.exp(dy) + 1)
        np.exp(dy, out=dy)
        np.add(dy, 1, out=dy)
        np.reciprocal(dy, out=dy)
        return dy
