from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array


class KLDivergenceMetric[T: Array](Metric[T]):
    format = "kld: %.7f"
