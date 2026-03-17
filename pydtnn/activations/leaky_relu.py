import logging
logger = logging.getLogger(__name__)

from pydtnn.activations.relu import Relu
from pydtnn.utils.constants import ArrayShape, Array

class LeakyRelu[T: Array](Relu[T]):

    def __init__(self, shape: ArrayShape = (1,), negative_slope: float = 0.01):
        super().__init__(shape)
        self.negative_slope: float = negative_slope
