import logging

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array

logger = logging.getLogger(__name__)


class CategoricalHinge[T: Array](Metric[T]):
    format = "hin: %.7f"
