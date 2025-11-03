from pydtnn.metrics.confusion_matrix import ConfusionMatrix
from pydtnn.utils.types import Array


class BinaryConfusionMatrix[T: Array](ConfusionMatrix[T]):
    conf_matrix: T = None  # type: ignore
