import logging
logger = logging.getLogger(__name__)

from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array


class CategoricalMSE[T:Array](Metric[T]):
    format = "mse: %.7f"
