"""Numpy backend implementation for metrics in the PyDTNN framework."""

import logging
from typing import TYPE_CHECKING

from pydtnn.libs import numpy as np
from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.metrics.abstract.metric import Metric

__all__ = ("MetricNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class MetricNumpy(Metric[np.ndarray], BaseNumpy):
    """Extends a Metric class with the attributes and methods required by CPU Metrics."""
