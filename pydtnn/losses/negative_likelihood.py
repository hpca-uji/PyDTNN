"""Negative Log Likelihood Loss loss implementation for the PyDTNN framework."""

import logging

from pydtnn.losses.abstract.loss import Loss
from pydtnn.utils.constants import Array

__all__ = ("NegativeLikelihood",)

logger = logging.getLogger(__name__)


class NegativeLikelihood[T: Array](Loss[T]):  # noqa: D101 (generics not detected)
    """Computes the categorical cross-entropy loss between predictions and targets."""
