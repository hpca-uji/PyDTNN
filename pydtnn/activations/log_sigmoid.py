"""Log activation module for PyDTNN."""

import logging

from pydtnn.activations.abstract.activation import Activation
from pydtnn.utils.constants import Array

__all__ = ("LogSigmoid",)

logger = logging.getLogger(__name__)


class LogSigmoid[T: Array](Activation[T]):  # noqa: D101 (generics not detected)
    """LogSigmoid activation function implementation."""
