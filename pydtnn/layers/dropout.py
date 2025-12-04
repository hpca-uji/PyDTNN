from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array
from pydtnn.utils.constants import ArrayShape


class Dropout[T: Array](Layer[T]):
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = min(1., max(0., rate))

    def initialize(self, prev_shape: ArrayShape, x: T | None):
        super().initialize(prev_shape, x)
        self.shape = prev_shape
