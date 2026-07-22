"""Module for computing and managing binary confusion matrices in PyDTNN."""

import logging

from pydtnn.metrics.confusion_matrix import ConfusionMatrix
from pydtnn.utils.constants import Array

__all__ = ("BinaryConfusionMatrix",)

logger = logging.getLogger(__name__)


class BinaryConfusionMatrix[T: Array](ConfusionMatrix[T]):  # noqa: D101 (generics not detected)
    """A confusion matrix implementation specifically for binary classification tasks."""
