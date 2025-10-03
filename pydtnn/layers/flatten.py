from abc import ABC

from .layer import Layer
from ..performance_models import *


class Flatten(Layer, ABC):

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.shape = (int(np.prod(prev_shape)),)
