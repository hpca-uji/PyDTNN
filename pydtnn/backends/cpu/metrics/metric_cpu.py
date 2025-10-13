from abc import ABC

from pydtnn.metrics import Metric

from numpy import ndarray

class MetricCPU(Metric[ndarray], ABC):
    """
    Extends a Metric class with the attributes and methods required by CPU Metrics.
    """
