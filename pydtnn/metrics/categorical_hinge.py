from pydtnn.metrics.metric import Metric
from pydtnn.utils.types import Array


class CategoricalHinge[T: Array](Metric[T]):
    format = "hin: %.7f"
