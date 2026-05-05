import logging

from pydtnn.activations.activation import Activation
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("Softmax",)

logger = logging.getLogger(__name__)


class Softmax[T: Array](Activation[T]):
    def __init__(self, shape: ArrayShape = (1,), axis: int = 1):
        super().__init__(shape)
        self.axis_dim = axis
