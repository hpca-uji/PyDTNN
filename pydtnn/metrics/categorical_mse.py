import logging

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("CategoricalMSE",)

logger = logging.getLogger(__name__)


class CategoricalMSE[T: Array](Metric[T]):
    format = "mse: %.7f"
