from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array


class RegressionMSE[T: Array](Metric[T]):
    format = "mse: %.7f"
