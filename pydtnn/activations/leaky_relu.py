import logging

from pydtnn.activations.relu import Relu
from pydtnn.utils.constants import Array, ArrayShape

__all__ = (
    "LeakyRelu",
)

logger = logging.getLogger(__name__)


class LeakyRelu[T: Array](Relu[T]):
    def __init__(self, shape: ArrayShape = (1,), negative_slope: float = 0.01):
        super().__init__(shape)
        self.negative_slope: float = negative_slope
