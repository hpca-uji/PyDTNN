from pydtnn.activations.relu import Relu
from pydtnn.utils.types import shape_t

class LeakyRelu(Relu):

    def __init__(self, shape: shape_t = (1,), negative_slope: float = 0.01):
        super().__init__(shape)
        self.negative_slope: float = negative_slope
