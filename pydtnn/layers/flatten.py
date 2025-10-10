from abc import ABC

from pydtnn.layers.layer import Layer
from pydtnn.performance_models import *

from pydtnn.utils.types import Array

class Flatten(Layer, ABC):

    def initialize(self, prev_shape: tuple[int, ...], x: Array | None = None):
        super().initialize(prev_shape, x)
        self.shape = (int(np.prod(prev_shape)),)
