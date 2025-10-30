from pydtnn.metrics.metric import Metric
from pydtnn.utils.types import Array


class CategoricalMSE[T:Array](Metric[T]):
    format = "mse: %.7f"
