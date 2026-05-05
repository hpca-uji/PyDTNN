import logging

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("KLDivergenceMetric",)

logger = logging.getLogger(__name__)


class KLDivergenceMetric[T: Array](Metric[T]):
    format = "kld: %.7f"
