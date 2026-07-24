"""NumPy backend implementation of the Rectified Linear Unit (ReLU) activation function."""

import logging
from typing import TYPE_CHECKING

from pydtnn.activations.relu import Relu
from pydtnn.backends.numpy.activations.abstract.activation import ActivationNumpy
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape

__all__ = ("ReluNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class ReluNumpy(Relu[np.ndarray], ActivationNumpy):
    """NumPy-based ReLU activation layer."""

    def __init__(self, shape: ArrayShape = (1,)) -> None:
        """Initializes the ReLU layer with a specific shape."""
        super().__init__(shape)
        self.mask: np.ndarray = None

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initializes internal buffers for forward and backward passes."""
        super()._model_init(prev_shape, x)
        # NOTE: These attributes only store data, their value before the operation
        # doesn't matter; they're initalized due avoid warnings in
        # "LayerAndActivationBase.export".
        self._y = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)
        self._mask = np.zeros((self.model.batch_size, *self.prev_shape), dtype=np.int8)

        self.memory_used += self._y.nbytes
        self.memory_used += self._mask.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the forward pass of the ReLU activation."""
        self.y = self._y[: x.shape[0], :]
        self.mask = self._mask[: x.shape[0], :]

        np.clip(x, 0, None, out=self.y)
        np.greater(x, 0, out=self.mask)

        self.y = np.asarray(self.y, dtype=self.model.dtype, order="C")
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Computes the backward pass of the ReLU activation."""
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype)
        return np.asarray(dy, dtype=self.model.dtype, order="C")
