"""
Regression Mean Absolute Error (MAE) metric implementation.
"""
import logging

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("RegressionMAE",)

logger = logging.getLogger(__name__)


class RegressionMAE[T: Array](Metric[T]):
    """
    Computes the Mean Absolute Error (MAE) for regression tasks.
    """
    format = "mae: %.7f"