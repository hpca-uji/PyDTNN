import logging

from pydtnn.metrics.confusion_matrix import ConfusionMatrix
from pydtnn.utils.constants import Array

__all__ = ("BinaryConfusionMatrix",)

logger = logging.getLogger(__name__)


class BinaryConfusionMatrix[T: Array](ConfusionMatrix[T]):
    conf_matrix: T = None  # type: ignore
