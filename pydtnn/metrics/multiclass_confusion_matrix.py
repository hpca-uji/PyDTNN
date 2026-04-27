import logging

from pydtnn.metrics.confusion_matrix import ConfusionMatrix
from pydtnn.utils.constants import Array

logger = logging.getLogger(__name__)


class MulticlassConfusionMatrix[T: Array](ConfusionMatrix[T]):
    pass
