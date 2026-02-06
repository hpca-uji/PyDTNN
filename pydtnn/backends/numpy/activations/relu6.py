from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
from pydtnn.backends.numpy.utils.relu_cython import capped_relu_cython
from pydtnn.activations.relu6 import Relu6
from pydtnn.backends.numpy.activations.activation import ActivationNumpy


class Relu6Numpy(Relu6[np.ndarray], ActivationNumpy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask: np.ndarray = None  # type: ignore (will be initalized in "initialize")

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        # NOTE: These attributes only store data, their value before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)
        self._mask = np.zeros((self.model.batch_size, *self.prev_shape), dtype=np.int8)

        self.real_memory_size += self._y.nbytes
        self.real_memory_size += self._mask.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y: np.ndarray = self._y[:x.shape[0], :]
        self.mask: np.ndarray = self._mask[:x.shape[0], :]

        np.clip(x, 0, self.cap, out=self.y)
        np.greater(x, 0, out=self.mask, dtype=np.int8)

        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # return dy * self.mask
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype)
        return dy
