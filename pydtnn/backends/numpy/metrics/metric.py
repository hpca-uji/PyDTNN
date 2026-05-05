import logging

from numpy import ndarray

from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.metrics.metric import Metric

__all__ = ("MetricNumpy",)

logger = logging.getLogger(__name__)


class MetricNumpy(Metric[ndarray], BaseNumpy):
    """
    Extends a Metric class with the attributes and methods required by CPU Metrics.
    """
