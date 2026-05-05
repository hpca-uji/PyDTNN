import logging

from pydtnn.activations.activation import Activation
from pydtnn.utils.constants import Array

__all__ = ("Sigmoid",)

logger = logging.getLogger(__name__)


class Sigmoid[T: Array](Activation[T]):
    pass
