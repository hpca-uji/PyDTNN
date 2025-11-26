import numpy as np
from pydtnn.layers.layer import Layer

from pydtnn.utils.constants import Array
from pydtnn.utils.constants import ArrayShape


class Flatten[T: Array](Layer[T]):

    def initialize(self, prev_shape: ArrayShape, x: T | None):
        super().initialize(prev_shape, x)
        self.shape = (int(np.prod(prev_shape)),)
