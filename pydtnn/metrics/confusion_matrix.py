"""Module for computing and managing confusion matrices in PyDTNN."""

import logging

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("ConfusionMatrix",)

logger = logging.getLogger(__name__)


class ConfusionMatrix[T: Array](Metric[T]):  # noqa: D101
    """A metric class for calculating the confusion matrix of classification predictions."""
