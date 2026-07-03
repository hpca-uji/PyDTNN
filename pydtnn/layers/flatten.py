"""Flatten layer for reshaping multi-dimensional input tensors into a 1D vector."""

import logging
import math

from pydtnn.layers.abstract.layer import Layer
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("Flatten",)

logger = logging.getLogger(__name__)


class Flatten[T: Array](Layer[T]):  # noqa: D101 (generics not detected)
    """A layer that flattens the input tensor into a single dimension."""

    def _model_init(self, prev_shape: ArrayShape, x: T | None) -> None:
        """
        Initializes the layer shape based on the product of the input dimensions.

        Args:
            prev_shape: The shape of the input tensor.
            x: Optional input data.
        """
        super()._model_init(prev_shape, x)
        self.shape = (int(math.prod(prev_shape)),)
