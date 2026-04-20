from pydtnn.utils.constants import Array
from pydtnn.metrics.metric import Metric
import logging
logger = logging.getLogger(__name__)


class KLDivergenceMetric[T: Array](Metric[T]):
    format = "kld: %.7f"
