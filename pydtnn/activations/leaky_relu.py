from pydtnn.activations.relu import Relu
from pydtnn.utils.types import ArrayShape

class LeakyRelu(Relu):

    def __init__(self, shape: ArrayShape = (1,), negative_slope: float = 0.01):
        super().__init__(shape)
        self.negative_slope: float = negative_slope
