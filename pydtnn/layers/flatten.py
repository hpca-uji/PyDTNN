from abc import ABC

from pydtnn.layers.layer import Layer
from pydtnn.performance_models import *


class Flatten(Layer, ABC):

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.shape = (int(np.prod(prev_shape)),)
