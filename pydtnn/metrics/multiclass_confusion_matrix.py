from pydtnn.utils.constants import Array
from pydtnn.metrics.confusion_matrix import ConfusionMatrix
import logging
logger = logging.getLogger(__name__)


class MulticlassConfusionMatrix[T: Array](ConfusionMatrix[T]):
    pass
