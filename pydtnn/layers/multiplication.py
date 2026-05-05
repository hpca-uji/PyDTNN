import logging

from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array

__all__ = (
    "Multiplication",
)

logger = logging.getLogger(__name__)


class Multiplication[T: Array](Layer[T]):
    pass
