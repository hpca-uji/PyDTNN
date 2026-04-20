from pydtnn.utils.constants import Array
from pydtnn.metrics.confusion_matrix import ConfusionMatrix
import logging
logger = logging.getLogger(__name__)


class BinaryConfusionMatrix[T: Array](ConfusionMatrix[T]):
    conf_matrix: T = None  # type: ignore
