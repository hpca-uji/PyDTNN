from abc import ABC

from pydtnn.layers.layer import Layer
from pydtnn.performance_models import *

from pydtnn.utils.types import Array
from pydtnn.utils.types import shape_t

class Flatten[T: Array](Layer, ABC):

    def initialize(self, prev_shape: shape_t, x: T | None = None):
        super().initialize(prev_shape, x)
        self.shape = (int(np.prod(prev_shape)),)
