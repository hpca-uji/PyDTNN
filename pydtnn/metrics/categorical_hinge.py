from pydtnn.utils.constants import Array
from pydtnn.metrics.metric import Metric
import logging
logger = logging.getLogger(__name__)


class CategoricalHinge[T: Array](Metric[T]):
    format = "hin: %.7f"
