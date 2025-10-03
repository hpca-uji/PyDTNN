import numpy as np

from pydtnn.activations.activation import Activation


class Tanh(Activation):

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.out = np.empty(shape=self.shape, dtype=self.model.dtype)
