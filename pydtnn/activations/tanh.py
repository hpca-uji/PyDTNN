import numpy as np

from pydtnn.activations.activation import Activation


class Tanh(Activation):

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
