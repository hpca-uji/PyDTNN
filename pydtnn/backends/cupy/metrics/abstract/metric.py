"""
CuPy-based metric implementations for the PyDTNN framework.
"""

import logging

from pydtnn.backends.numpy.metrics.abstract.metric import MetricNumpy

__all__ = ("MetricCupy",)

logger = logging.getLogger(__name__)


class MetricCupy(MetricNumpy):
    """
    Extends a Metric class with the attributes and methods required by CPU Metrics.
    """
