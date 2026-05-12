"""Numpy backend implementation of the Arctanh activation function."""
import logging
from typing import TYPE_CHECKING

from pydtnn.activations.arctanh import Arctanh
from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.libs import numpy as np

__all__ = ("ArctanhNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class ArctanhNumpy(Arctanh[np.ndarray], ActivationNumpy):
    """Numpy-based Arctanh activation layer."""
    def __init__(self, *args, **kwargs):
        """Initialize the ArctanhNumpy layer."""
        super().__init__(*args, **kwargs)

    def _model_init(self, prev_shape, x=None):
        """Initialize model parameters and allocate memory for output."""
        super()._model_init(prev_shape, x)
        # NOTE: This attribute only stores data, its value before the operation doesn't matters; it's initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype)

        self.memory_used += self._y.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward pass of the Arctanh activation."""
        self.y = self._y[: x.shape[0], :]
        np.arctanh(x, out=self.y, casting="unsafe", dtype=self.model.dtype)
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Compute the backward pass of the Arctanh activation."""
        # return 1 / (1 + dy ** 2)
        np.power(dy, 2, out=dy, casting="unsafe", dtype=self.model.dtype)
        np.add(dy, 1, out=dy, casting="unsafe", dtype=self.model.dtype)
        np.reciprocal(dy, out=dy, casting="unsafe", dtype=self.model.dtype)
        return dy