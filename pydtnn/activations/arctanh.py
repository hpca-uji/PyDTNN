"""Module providing the Arctanh activation function for the PyDTNN framework."""

import logging

from pydtnn.activations.abstract.activation import Activation
from pydtnn.utils.constants import Array

__all__ = ("Arctanh",)

logger = logging.getLogger(__name__)


class Arctanh[T: Array](Activation[T]):  # noqa: D101 (generics not detected)
    """Arctanh activation layer that computes the inverse hyperbolic tangent of the input."""
