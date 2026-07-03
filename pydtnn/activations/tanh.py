"""Hyperbolic tangent activation function module."""

import logging

from pydtnn.activations.abstract.activation import Activation
from pydtnn.utils.constants import Array

__all__ = ("Tanh",)

logger = logging.getLogger(__name__)


class Tanh[T: Array](Activation[T]):  # noqa: D101 (generics not detected)
    """Hyperbolic tangent activation layer."""
