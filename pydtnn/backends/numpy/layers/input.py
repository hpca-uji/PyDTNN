import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.layers.input import Input
from pydtnn.libs import numpy as np

__all__ = (
    "InputNumpy",
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class InputNumpy(Input[np.ndarray], LayerNumpy):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=self.model.dtype, order="C")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return np.asarray(dy, dtype=self.model.dtype, order="C")

    def _sync_x_y(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_batch = np.asarray(x_batch, dtype=self.model.dtype, order="C")
        y_batch = np.asarray(y_batch, dtype=self.model.dtype, order="C")
        return x_batch, y_batch
