from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array


class CategoricalMAE[T: Array](Metric[T]):
    format = "mae: %.7f"
