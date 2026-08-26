"""Categorical Mean Absolute Error metric implementation for PyDTNN."""

import logging

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("CategoricalMAE",)

logger = logging.getLogger(__name__)


class CategoricalMAE[T: Array](Metric[T]):  # noqa: D101 (generics not detected)
    """Computes the Mean Absolute Error for categorical data."""
