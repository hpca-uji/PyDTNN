"""Cross Entropy Loss loss implementation for the PyDTNN framework."""

import logging

from pydtnn.losses.abstract.loss import Loss
from pydtnn.utils.constants import Array

__all__ = ("CrossEntropy",)

logger = logging.getLogger(__name__)


class CrossEntropy[T: Array](Loss[T]):  # noqa: D101 (generics not detected)
    """Computes the cross-entropy loss between predictions and targets."""
