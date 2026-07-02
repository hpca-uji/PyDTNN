"""Multiclass confusion matrix implementation for PyDTNN."""

import logging

from pydtnn.metrics.confusion_matrix import ConfusionMatrix
from pydtnn.utils.constants import Array

__all__ = ("MulticlassConfusionMatrix",)

logger = logging.getLogger(__name__)


class MulticlassConfusionMatrix[T: Array](ConfusionMatrix[T]):  # noqa: D101
    """Computes and stores the confusion matrix for multiclass classification tasks."""
