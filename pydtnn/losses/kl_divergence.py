"""Kullback-Leibler Divergence loss implementation for PyDTNN."""

import logging

from pydtnn.losses.abstract.loss import Loss
from pydtnn.utils.constants import Array

__all__ = ("KLDivergence",)

logger = logging.getLogger(__name__)


class KLDivergence[T: Array](Loss[T]):
    """Computes the Kullback-Leibler divergence between two probability distributions."""

    format = "kld: %.7f"
