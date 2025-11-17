from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array


class CategoricalAccuracy[T: Array](Metric[T]):
    format = "acc: %5.2f%%"
