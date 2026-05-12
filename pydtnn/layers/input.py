"""
Input layer module for PyDTNN.
"""

import logging

import numpy as np

from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array

__all__ = ("Input",)

logger = logging.getLogger(__name__)


class Input[T: Array](Layer[T]):
    """
    Represents the input layer of a neural network.
    """

    def __init__(self, shape: tuple = (1,)):
        """
        Initializes the Input layer with a specific shape.

        Args:
            shape (tuple): The shape of the input data.
        """
        super().__init__(shape)

    def _sync_x_y(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[T, T]:
        """
        Synchronizes input and target batches.

        Args:
            x_batch (np.ndarray): The input batch.
            y_batch (np.ndarray): The target batch.

        Returns:
            tuple[T, T]: A tuple containing the input and target batches.
        """
        return (x_batch, y_batch)  # type: ignore (It's fine)
