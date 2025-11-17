from pydtnn.metrics.confusion_matrix import ConfusionMatrix
from pydtnn.utils.constants import Array


class BinaryConfusionMatrix[T: Array](ConfusionMatrix[T]):
    conf_matrix: T = None  # type: ignore
