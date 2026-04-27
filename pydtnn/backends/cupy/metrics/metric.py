import logging

from pydtnn.backends.numpy.metrics.metric import MetricNumpy

logger = logging.getLogger(__name__)


class MetricCupy(MetricNumpy):
    """
    Extends a Metric class with the attributes and methods required by CPU Metrics.
    """
