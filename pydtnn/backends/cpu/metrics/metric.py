from pydtnn.metrics.metric import Metric

from numpy import ndarray


class MetricCPU(Metric[ndarray]):
    """
    Extends a Metric class with the attributes and methods required by CPU Metrics.
    """
