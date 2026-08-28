"""Numpy backend implementation for the Input layer."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy
from pydtnn.layers.identity import Identity
from pydtnn.libs import numpy as np

__all__ = ("IdentityNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class IdentityNumpy(Identity[np.ndarray], LayerNumpy):
    """Numpy-based identity layer for handling data ingestion and type casting."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Passes the input through, ensuring it matches the model's dtype and memory layout."""
        return np.asarray(x, dtype=self.model.dtype, order="C")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Passes the gradient through, ensuring it matches the model's dtype and memory layout."""
        return np.asarray(dy, dtype=self.model.dtype, order="C")

    def _sync_x_y(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Synchronizes input and target batches to the model's dtype and memory layout."""
        x_batch = np.asarray(x_batch, dtype=self.model.dtype, order="C")
        y_batch = np.asarray(y_batch, dtype=self.model.dtype, order="C")
        return x_batch, y_batch
