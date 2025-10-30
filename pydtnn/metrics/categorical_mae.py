from pydtnn.metrics.metric import Metric
from pydtnn.utils.types import Array


class CategoricalMAE[T: Array](Metric[T]):
    format = "mae: %.7f"
