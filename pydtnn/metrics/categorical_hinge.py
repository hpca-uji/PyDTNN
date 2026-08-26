"""Categorical hinge loss metric implementation."""

import logging

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("CategoricalHinge",)

logger = logging.getLogger(__name__)


class CategoricalHinge[T: Array](Metric[T]):  # noqa: D101 (generics not detected)
    """Computes the categorical hinge loss between y_true and y_pred."""
