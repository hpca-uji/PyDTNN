"""Module for calculating Kullback-Leibler divergence metrics in PyDTNN."""

import logging

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.utils.constants import Array

__all__ = ("KLDivergenceMetric",)

logger = logging.getLogger(__name__)


class KLDivergenceMetric[T: Array](Metric[T]):
    """Metric class for computing the Kullback-Leibler divergence between two distributions."""

    format = "kld: %.7f"
