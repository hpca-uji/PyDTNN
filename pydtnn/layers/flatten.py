import numpy as np
from pydtnn.layers.layer import Layer

from pydtnn.utils.types import Array
from pydtnn.utils.types import ArrayShape


class Flatten[T: Array](Layer):

    def initialize(self, prev_shape: ArrayShape, x: T | None = None):
        super().initialize(prev_shape, x)
        self.shape = (int(np.prod(prev_shape)),)
