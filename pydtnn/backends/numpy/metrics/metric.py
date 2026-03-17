import logging
logger = logging.getLogger(__name__)

from pydtnn.metrics.metric import Metric

from numpy import ndarray


class MetricNumpy(Metric[ndarray]):
    """
    Extends a Metric class with the attributes and methods required by CPU Metrics.
    """
