import logging

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("RegressionMSE",)

logger = logging.getLogger(__name__)


class RegressionMSE[T: Array](Metric[T]):
    format = "mse: %.7f"
