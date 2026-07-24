"""Input layer module for PyDTNN."""

import logging

import numpy as np

from pydtnn.layers.abstract.layer import Layer
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("Input",)

logger = logging.getLogger(__name__)


class Input[T: Array](Layer[T]):  # noqa: D101 (generics not detected)
    """Represents the input layer of a neural network."""

    def __init__(self, shape: tuple = (1,)) -> None:
        """
        Initializes the Input layer with a specific shape.

        Args:
            shape (tuple): The shape of the input data.
        """
        super().__init__(shape)

    def _model_init(self, prev_shape: ArrayShape, x: T | None) -> None:
        """Initialize layer state within the model context."""
        super()._model_init(prev_shape or self.shape, x)

    def _sync_x_y(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[T, T]:
        """
        Synchronizes input and target batches.

        Args:
            x_batch (np.ndarray): The input batch.
            y_batch (np.ndarray): The target batch.

        Returns:
            tuple[T, T]: A tuple containing the input and target batches.
        """
        return (x_batch, y_batch)  # pyright: ignore[reportReturnType]
