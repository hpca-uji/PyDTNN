from pydtnn.utils.constants import Array
from pydtnn.metrics.metric import Metric
import logging
logger = logging.getLogger(__name__)


class RegressionMAE[T: Array](Metric[T]):
    format = "mae: %.7f"
