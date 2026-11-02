"""Input layer module for PyDTNN."""

import logging

import numpy as np

from pydtnn.layers.abstract.layer import Layer
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("Identity",)

logger = logging.getLogger(__name__)


class Identity[T: Array](Layer[T]):  # noqa: D101 (generics not detected)
    """Represents the identity layer of a neural network."""

    def __init__(self, shape: tuple = ()) -> None:
        """
        Initializes the Input layer with a specific shape.

        Args:
            shape (tuple): The shape of the input data.
        """
        super().__init__(shape)

    def _model_init(self, prev_shape: ArrayShape, x: T | None) -> None:
        """Initialize layer state within the model context."""
        super()._model_init(prev_shape or self.shape, x)
        self.shape = self.prev_shape
