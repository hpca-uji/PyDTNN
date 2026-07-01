"""Categorical Mean Squared Error metric implementation for PyDTNN."""

import logging

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("CategoricalMSE",)

logger = logging.getLogger(__name__)


class CategoricalMSE[T: Array](Metric[T]):
    """Computes the Mean Squared Error between categorical predictions and targets."""

    format = "mse: %.7f"
