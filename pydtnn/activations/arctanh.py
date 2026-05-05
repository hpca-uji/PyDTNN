import logging

from pydtnn.activations.activation import Activation
from pydtnn.utils.constants import Array

__all__ = ("Arctanh",)

logger = logging.getLogger(__name__)


class Arctanh[T: Array](Activation[T]):
    pass
