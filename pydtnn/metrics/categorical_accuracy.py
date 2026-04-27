import logging

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array

logger = logging.getLogger(__name__)


class CategoricalAccuracy[T: Array](Metric[T]):
    format = "acc: %5.2f%%"
