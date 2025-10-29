from abc import ABC

from pydtnn.metrics.metric import Metric
from pydtnn.utils.types import Array

class ConfusionMatrix[T: Array](Metric[T], ABC):
    pass