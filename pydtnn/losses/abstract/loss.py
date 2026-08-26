"""
Loss module for PyDTNN.

Provides the base class for loss functions and utility functions for selecting
loss implementations.
"""

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from pydtnn.abstract.base import Base
from pydtnn.utils.constants import Array

if TYPE_CHECKING:
    from pydtnn.model.base import Base as Model

__all__ = ("Loss",)

logger = logging.getLogger(__name__)


class Loss[T: Array](Base):  # noqa: D101 (generics not detected)
    """
    Base class for all loss functions in PyDTNN.

    Attributes:
        eps (float): Small value to prevent numerical instability.
        format (str): Format string for loss representation.
    """

    def format(self, value: Any) -> str:
        name = re.sub("[^A-Z]", "", self.canonical_name).lower()
        if isinstance(value, (float, np.floating)):
            return f"{name}: {value:.5f}"
        else:
            return f"{name}: {value}"

    def _weights_to_tensor(self, weights: list[float]) -> np.ndarray:
        w = None
        # NOTE: This may not work very well in case self.model.dtype is an int
        if weights:
            w = np.ascontiguousarray(weights, dtype=self.model.dtype)
        else:
            w = np.ones(self.model.output_shape, dtype=self.model.dtype, order="C")
        return w

    def __init__(self, eps: float = 1e-8) -> None:
        """
        Initializes the Loss instance.

        Args:
            eps (float): Epsilon value for numerical stability. Defaults to 1e-8.
        """
        super().__init__()
        self.eps = eps

    def _model_init(self) -> None:
        """Initializes model-specific parameters for the loss function."""

        super()._model_init()
        self.shape = (self.model.batch_size, *self.model.output_shape)
        if self.model.use_class_weights:
            if self.model.class_weights:
                weights = list(self.model.class_weights)
            else:
                weights = self.model.dataset.class_weights
        else:
            weights = []

        self.weights = self._weights_to_tensor(weights)

    def compute(self, y_pred: T, y_targ: T) -> tuple[float, T]:
        """
        Computes the loss value and the gradient.

        Args:
            y_pred (T): Predicted values.
            y_targ (T): Target values.

        Returns:
            tuple[float, T]: A tuple containing the scalar loss and the gradient.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError()

    @classmethod
    def from_model[Y: Loss](cls: type[Y], model: Model[T]) -> Y:
        """
        Create an CategoricalCrossEntropy instance from a model configuration.

        Args:
            model: The model instance to extract parameters from.

        Returns:
            An initialized CategoricalCrossEntropy optimizer.
        """
        return cls(eps=model.loss_eps)
