"""Log activation module for PyDTNN."""

import logging

from pydtnn.activations.abstract.activation import Activation
from pydtnn.activations.softmax import Softmax
from pydtnn.utils.constants import Array

__all__ = ("LogSoftmax",)

logger = logging.getLogger(__name__)


class LogSoftmax[T: Array](Softmax[T], Activation[T]):  # noqa: D101 (generics not detected)
    """LogSoftmax activation function implementation."""
