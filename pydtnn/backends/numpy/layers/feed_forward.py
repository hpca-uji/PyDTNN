"""Numpy backend implementation of the FeedForward layer."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.block_layer import AbstractBlockLayerNumpy
from pydtnn.layers.feed_forward import FeedForward
from pydtnn.libs import numpy as np

__all__ = ("FeedForwardNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class FeedForwardNumpy(FeedForward[np.ndarray], AbstractBlockLayerNumpy):
    """Numpy-based FeedForward layer implementation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the forward pass through the feed-forward network."""
        # NOTE: It should be only one path.
        for layer in self.paths[0]:
            x = layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the feed-forward network."""
        # NOTE: It should be only one path.
        dx = dy
        for layer in reversed(self.paths[0]):
            dx = layer.backward(dx)
        return dx
