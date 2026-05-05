import logging

from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array, ArrayShape

__all__ = (
    "Scalar",
)

logger = logging.getLogger(__name__)


class Scalar[T: Array](Layer[T]):
    def __init__(self, shape: ArrayShape = (1,), scale: float = 1.0):
        super().__init__(shape)
        self.scale = scale
