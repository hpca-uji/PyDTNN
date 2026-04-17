import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.metrics.metric import Metric

from numpy import ndarray


class MetricNumpy(Metric[ndarray], BaseNumpy):
    """
    Extends a Metric class with the attributes and methods required by CPU Metrics.
    """
