from numpy import ndarray
from pydtnn.metrics.metric import Metric
from pydtnn.backends.numpy.abstract.base import BaseNumpy
import logging
logger = logging.getLogger(__name__)


class MetricNumpy(Metric[ndarray], BaseNumpy):
    """
    Extends a Metric class with the attributes and methods required by CPU Metrics.
    """
