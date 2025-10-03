from abc import ABC

from pydtnn.metrics import Metric


class MetricCPU(Metric, ABC):
    """
    Extends a Metric class with the attributes and methods required by CPU Metrics.
    """
