"""
Regression Mean Squared Error metric implementation.
"""

import logging

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("RegressionMSE",)

logger = logging.getLogger(__name__)


class RegressionMSE[T: Array](Metric[T]):
    """
    Computes the Mean Squared Error (MSE) for regression tasks.
    """

    format = "mse: %.7f"
