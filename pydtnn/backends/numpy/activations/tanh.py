"""NumPy backend implementation of the Tanh activation function."""

import logging
from typing import TYPE_CHECKING

from pydtnn.activations.tanh import Tanh
from pydtnn.backends.numpy.activations.abstract.activation import ActivationNumpy
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape

__all__ = ("TanhNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class TanhNumpy(Tanh[np.ndarray], ActivationNumpy):
    """NumPy-based Tanh activation layer."""

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initialize the layer model and allocate memory for the output buffer."""
        super()._model_init(prev_shape, x)
        # NOTE: This attribute only stores data, its value before the operation
        # doesn't matters; it's initalized due avoid warnings in
        # "LayerAndActivationBase.export".
        self._y = np.zeros((self.model.batch_size, *prev_shape), dtype=self.model.dtype)

        self.memory_used += self._y.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward pass of the Tanh activation."""
        self.y = self._y[: x.shape[0], :]
        np.tanh(x, out=self.y, casting="unsafe", dtype=self.model.dtype)
        self.y = np.asarray(self.y, dtype=self.model.dtype, order="C")
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Compute the backward pass of the Tanh activation."""
        # return 1 - np.tanh(dy) ** 2
        np.tanh(dy, out=dy, casting="unsafe", dtype=dy.dtype)
        np.power(dy, 2, out=dy, dtype=self.model.dtype)
        np.subtract(1, dy, out=dy, dtype=self.model.dtype)

        return np.asarray(dy, dtype=self.model.dtype, order="C")
