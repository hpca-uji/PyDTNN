"""Regression Mean Absolute Error (MAE) metric implementation."""

import logging

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("RegressionMAE",)

logger = logging.getLogger(__name__)


class RegressionMAE[T: Array](Metric[T]):  # noqa: D101 (generics not detected)
    """Computes the Mean Absolute Error (MAE) for regression tasks."""
