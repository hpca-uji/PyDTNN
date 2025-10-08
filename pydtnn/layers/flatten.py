from abc import ABC

from pydtnn.layers.layer import Layer, Array
from pydtnn.performance_models import *


class Flatten(Layer, ABC):

    def initialize(self, prev_shape: tuple[int, ...], x: Array | None = None):
        super().initialize(prev_shape, x)
        self.shape = (int(np.prod(prev_shape)),)
