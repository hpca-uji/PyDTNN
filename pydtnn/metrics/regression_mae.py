import logging

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("RegressionMAE",)

logger = logging.getLogger(__name__)


class RegressionMAE[T: Array](Metric[T]):
    format = "mae: %.7f"
