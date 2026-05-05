import logging

from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array

__all__ = (
    "Input",
)

logger = logging.getLogger(__name__)


class Input[T: Array](Layer[T]):
    def __init__(self, shape: tuple = (1,)):
        super().__init__(shape)
