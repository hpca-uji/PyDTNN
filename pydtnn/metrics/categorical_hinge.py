"""
Categorical hinge loss metric implementation.
"""
import logging

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("CategoricalHinge",)

logger = logging.getLogger(__name__)


class CategoricalHinge[T: Array](Metric[T]):
    """
    Computes the categorical hinge loss between y_true and y_pred.
    """
    format = "hin: %.7f"