"""
Module for computing and managing confusion matrices in PyDTNN.
"""

import logging

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("ConfusionMatrix",)

logger = logging.getLogger(__name__)


class ConfusionMatrix[T: Array](Metric[T]):
    """
    A metric class for calculating the confusion matrix of classification predictions.
    """

    pass
