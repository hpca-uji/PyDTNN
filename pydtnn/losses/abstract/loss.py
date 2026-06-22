"""
Loss module for PyDTNN.

Provides the base class for loss functions and utility functions for selecting
loss implementations.
"""

import logging

import numpy as np

from pydtnn.abstract.base import Base
from pydtnn.utils.constants import Array

__all__ = ("Loss",)

logger = logging.getLogger(__name__)


class Loss[T: Array](Base):
    """
    Base class for all loss functions in PyDTNN.

    Attributes:
        eps (float): Small value to prevent numerical instability.
        format (str): Format string for loss representation.
    """

    format = ""

    def _weights_to_tensor(self, weights: list[float] | None) -> np.ndarray:
        w = None
        dtype = np.float32
        if weights is not None:
            w = np.ascontiguousarray(weights, dtype=dtype)
        else:
            w = np.ones(self.model.output_shape, dtype=dtype, order="C")
        return w

    def __init__(self, weights: list[float] | None = None, eps: float = 1e-8):
        """
        Initializes the Loss instance.

        Args:
            eps (float): Epsilon value for numerical stability. Defaults to 1e-8.
        """
        super().__init__()
        self.eps = eps
        self.weights = weights  # type: ignore (It will be initalized later)

    def _model_init(self) -> None:
        """
        Initializes model-specific parameters for the loss function.
        """
        super()._model_init()
        self.shape = (self.model.batch_size, *self.model.output_shape)
        self.weights: T = self._weights_to_tensor(self.weights) # type: ignore (It will be initalized here)

    def compute(self, y_pred: T, y_targ: T, batch_size: int) -> tuple[float, T]:
        """
        Computes the loss value and the gradient.

        Args:
            y_pred (T): Predicted values.
            y_targ (T): Target values.
            batch_size (int): Size of the current batch.

        Returns:
            tuple[float, T]: A tuple containing the scalar loss and the gradient.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError()
