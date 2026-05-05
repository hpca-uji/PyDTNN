import logging

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("ConfusionMatrix",)

logger = logging.getLogger(__name__)


class ConfusionMatrix[T: Array](Metric[T]):
    pass
