import warnings
from abc import ABC

from pydtnn.layers.layer import Layer

from pydtnn.utils.types import Array


class Input(Layer, ABC):

    def __init__(self, shape: tuple = (1,)):
        super().__init__(shape)

    def initialize(self, prev_shape: tuple, x: Array | None = None):
        super().initialize(prev_shape, x)
