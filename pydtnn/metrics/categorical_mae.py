"""
Categorical Mean Absolute Error metric implementation for PyDTNN.
"""

import logging

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("CategoricalMAE",)

logger = logging.getLogger(__name__)


class CategoricalMAE[T: Array](Metric[T]):
    """
    Computes the Mean Absolute Error for categorical data.
    """

    format = "mae: %.7f"
