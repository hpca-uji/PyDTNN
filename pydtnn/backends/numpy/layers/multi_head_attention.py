"""Multi-head attention layer implementation for the NumPy backend."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.block_layer import AbstractBlockLayerNumpy
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.libs import numpy as np

__all__ = ("MultiHeadAttentionNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class MultiHeadAttentionNumpy(MultiHeadAttention[np.ndarray], AbstractBlockLayerNumpy):
    """NumPy implementation of the Multi-Head Attention mechanism."""

    def _split(self, x: np.ndarray) -> np.ndarray:
        """Method to split the data in self.heads parts."""
        x_splitted = np.split(x, self.heads, axis=0)
        x = np.array(x_splitted, dtype=np.object_, ndmax=1)
        return x

    def _concat(self, dx: np.ndarray) -> np.ndarray:
        """Method get the data into the original shape."""
        return np.concat(dx)
