from abc import ABC


from pydtnn.layers.layer import Layer
from pydtnn.utils.types import Array
from pydtnn.utils.types import ArrayShape

class Dropout[T: Array](Layer, ABC):

    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = min(1., max(0., rate))

    def initialize(self, prev_shape: ArrayShape, x: T | None = None):
        super().initialize(prev_shape, x)
        self.shape = prev_shape

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^37s}|".format("", "rate=%.2f" % self.rate))
