"""
Hyperbolic tangent activation function module.
"""
import logging

from pydtnn.activations.activation import Activation
from pydtnn.utils.constants import Array

__all__ = ("Tanh",)

logger = logging.getLogger(__name__)


class Tanh[T: Array](Activation[T]):
    """
    Hyperbolic tangent activation layer.
    """
    pass