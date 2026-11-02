"""Numpy backend implementation for layerable components in PyDTNN."""

import numpy as np

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.numpy.abstract.base import BaseNumpy

__all__ = ("LayerableNumpy",)


class LayerableNumpy(Layerable[np.ndarray], BaseNumpy):
    """Numpy-specific implementation of a layerable component supporting distributed operations."""

    def _sync_x_y(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Synchronizes input and target batches to the model's dtype and memory layout."""
        x_batch = np.asarray(x_batch, dtype=self.model.dtype, order="C")
        y_batch = np.asarray(y_batch, dtype=self.model.dtype, order="C")
        return x_batch, y_batch
