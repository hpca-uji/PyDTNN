"""Regression Mean Squared Error metric implementation."""

import logging

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("RegressionMSE",)

logger = logging.getLogger(__name__)


class RegressionMSE[T: Array](Metric[T]):  # noqa: D101 (generics not detected)
    """Computes the Mean Squared Error (MSE) for regression tasks."""
