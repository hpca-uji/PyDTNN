import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.metrics.metric import MetricNumpy

class MetricCupy(MetricNumpy):
    """
    Extends a Metric class with the attributes and methods required by CPU Metrics.
    """
