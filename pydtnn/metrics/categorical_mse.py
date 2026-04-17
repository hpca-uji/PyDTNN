from pydtnn.utils.constants import Array
from pydtnn.metrics.metric import Metric
import logging
logger = logging.getLogger(__name__)


class CategoricalMSE[T:Array](Metric[T]):
    format = "mse: %.7f"
