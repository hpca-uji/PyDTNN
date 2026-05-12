"""
Categorical Cross-Entropy loss implementation for the PyDTNN framework.
"""

import logging

from pydtnn.losses.loss import Loss
from pydtnn.utils.constants import Array

__all__ = ("CategoricalCrossEntropy",)

logger = logging.getLogger(__name__)


class CategoricalCrossEntropy[T: Array](Loss[T]):
    """
    Computes the categorical cross-entropy loss between predictions and targets.
    """

    format = "cce: %.7f"
