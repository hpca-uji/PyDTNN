from pydtnn.activations.relu import Relu
from pydtnn.utils.types import ArrayShape

# NOTE -> "CappedRelu": https://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf


class Relu6(Relu):
    # NOTE: This is a ReLU6 *iif* cap is 6, but it's more interesting a implementation where the user have the freedom to choose their cap.
    def __init__(self, shape: ArrayShape = (1,), cap: float = 6.0):
        super().__init__(shape)
        self.cap: float = cap
