"""Categorical accuracy metric implementation for PyDTNN."""

import logging

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("CategoricalAccuracy",)

logger = logging.getLogger(__name__)


class CategoricalAccuracy[T: Array](Metric[T]):  # noqa: D101 (generics not detected)
    """Metric to calculate the categorical accuracy of model predictions."""

    format = "acc: %5.2f%%"
