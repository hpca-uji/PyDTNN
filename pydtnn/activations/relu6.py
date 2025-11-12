from pydtnn.activations.relu import Relu
from pydtnn.utils.types import ArrayShape, Array
# NOTE -> "CappedRelu": https://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf
from typing import Self
class Relu6[T: Array](Relu[T]):
    # NOTE: This is a ReLU6 *iif* cap is 6, but it's more interesting a implementation where the user have the freedom to choose their cap.
    def __init__(self, shape: ArrayShape = (1,), cap: float = 6.0):
        super().__init__(shape)
        self.cap: float = cap

    def copy_from(self, other: Self) -> None:
        super().copy_from(other)
        self.cap = other.cap
