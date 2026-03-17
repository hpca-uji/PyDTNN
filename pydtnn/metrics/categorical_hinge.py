import logging
logger = logging.getLogger(__name__)

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array


class CategoricalHinge[T: Array](Metric[T]):
    format = "hin: %.7f"
