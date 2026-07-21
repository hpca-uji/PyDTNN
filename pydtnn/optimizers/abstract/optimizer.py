"""
Optimizer module for PyDTNN.

This module provides the base class for all optimization algorithms and a utility
function to dynamically select optimizers by name.
"""

import logging

import numpy as np

from pydtnn.abstract.base import Base
from pydtnn.abstract.layerable import Layerable
from pydtnn.utils.constants import Array

__all__ = ("Optimizer",)

logger = logging.getLogger(__name__)


class Optimizer[T: Array](Base):  # noqa: D101 (generics not detected)
    """
    Optimizer abstract base class for updating model parameters.

    Attributes:
        learning_rate (float): The step size used for parameter updates.
        context (dict): Internal state storage for optimizer-specific parameters.
    """

    def __init__(self, learning_rate: float = 1e-2) -> None:
        """
        Initializes the optimizer with a learning rate.

        Args:
            learning_rate (float): The learning rate for the optimizer.
        """
        super().__init__()
        self.learning_rate: float = learning_rate
        self.context = dict[int, dict[str, int | T]]()

    def _model_init(self, layers: list[Layerable[T]]) -> None:
        """
        Initializes the optimizer with the model layers.

        Args:
            layers (list[Layerable[T]]): List of layers to be optimized.
        """
        super()._model_init()
        self.layers = layers

    @property
    def dtype(self) -> np.dtype:
        """
        Returns the data type of the model parameters.

        Returns:
            np.dtype: The numpy data type.
        """
        return self.model.dtype

    @property
    def gpudirect(self) -> bool:
        """
        Checks if the model supports GPU direct operations.

        Returns:
            bool: True if GPU direct is enabled, False otherwise.
        """
        return self.model.gpudirect

    def update(self, layer: Layerable) -> None:
        """
        Updates the parameters of the given layer.

        Args:
            layer (Layerable): The layer to update.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("method update of an Optimizer's child class is not implemented")
