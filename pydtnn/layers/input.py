import warnings
from abc import ABC

from .layer import Layer
from pydtnn.utils import encode_tensor


class Input(Layer, ABC):
    # NOTE: Input(shape) is expected to be in NHWC

    def __init__(self, shape: tuple = (1,)):
        super().__init__(shape)

    def initialize(self, prev_shape: tuple):
        super().initialize(prev_shape)
