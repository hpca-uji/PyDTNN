from abc import ABC


from .layer import Layer


class Dropout(Layer, ABC):

    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = min(1., max(0., rate))

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.shape = prev_shape

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^37s}|".format("", "rate=%.2f" % self.rate))
