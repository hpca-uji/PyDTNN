"""NumPy backend implementation of the ReLU6 activation function."""
import logging
from typing import TYPE_CHECKING

from pydtnn.activations.relu6 import Relu6
from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.libs import numpy as np

__all__ = ("Relu6Numpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class Relu6Numpy(Relu6[np.ndarray], ActivationNumpy):
    """NumPy implementation of the ReLU6 activation layer."""
    def __init__(self, *args, **kwargs):
        """Initialize the Relu6Numpy layer."""
        super().__init__(*args, **kwargs)
        self.mask: np.ndarray = None  # type: ignore (will be initalized in "initialize")

    def _model_init(self, prev_shape, x=None):
        """Initialize internal buffers for forward and backward passes."""
        super()._model_init(prev_shape, x)
        # NOTE: These attributes only store data, their value before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)
        self._mask = np.zeros((self.model.batch_size, *self.prev_shape), dtype=np.int8)

        self.memory_used += self._y.nbytes
        self.memory_used += self._mask.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass of ReLU6."""
        self.y: np.ndarray = self._y[: x.shape[0], :]
        self.mask: np.ndarray = self._mask[: x.shape[0], :]

        np.clip(x, 0, self.cap, out=self.y)
        np.greater(x, 0, out=self.mask, dtype=np.int8)

        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Perform the backward pass of ReLU6."""
        # return dy * self.mask
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype)
        return dy