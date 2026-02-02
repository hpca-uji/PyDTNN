from pydtnn.metrics.metric import Metric

from cupy import ndarray


class MetricCUPY(Metric[ndarray]):
    """
    Extends a Metric class with the attributes and methods required by CUPY Metrics.
    """
