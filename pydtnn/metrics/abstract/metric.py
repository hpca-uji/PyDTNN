"""
Metric module for PyDTNN.

This module provides the base class for defining evaluation metrics and a utility
function for dynamically selecting metric implementations.
"""

import logging
from abc import abstractmethod

import numpy as np

from pydtnn.abstract.base import Base
from pydtnn.utils.constants import Array

__all__ = ("Metric",)

logger = logging.getLogger(__name__)


class Metric[T: Array](Base):
    """
    Abstract base class for all evaluation metrics in PyDTNN.

    Attributes:
        format (str): String format for metric display.
        order (int): Execution order priority.
        eps (float): Small epsilon value for numerical stability.
    """

    format = ""
    order = 0  # No need of special order.

    def __init__(self, eps=1e-8):
        """
        Initializes the metric with a stability constant.

        Args:
            eps (float): Epsilon value to prevent division by zero or log errors.
        """
        super().__init__()
        self.eps = eps

    def _model_init(self) -> None:
        """
        Initializes metric-specific properties based on the associated model.
        """
        self.dtype: np.dtype = (
            np.dtype(np.float32) if np.issubdtype(self.model.dtype, np.int32) else self.model.dtype
        )
        # NOTE: self.dtype is necessary before calling super.
        super()._model_init()
        self.shape = (self.model.batch_size, *self.model.output_shape)

    @abstractmethod
    def compute(self, y_pred: T, y_targ: T) -> float | np.ndarray:
        """
        Computes the metric value given predictions and targets.

        Args:
            y_pred (T): Predicted values from the model.
            y_targ (T): Ground truth target values.

        Returns:
            float | np.ndarray: The computed metric result.
        """
        pass
