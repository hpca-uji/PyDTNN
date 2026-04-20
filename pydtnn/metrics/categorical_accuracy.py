from pydtnn.utils.constants import Array
from pydtnn.metrics.metric import Metric
import logging
logger = logging.getLogger(__name__)


class CategoricalAccuracy[T: Array](Metric[T]):
    format = "acc: %5.2f%%"
