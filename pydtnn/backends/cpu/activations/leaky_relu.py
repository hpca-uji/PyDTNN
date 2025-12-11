from pydtnn.backends.cpu.utils.relu_cython import leaky_relu_cython
from pydtnn.activations.leaky_relu import LeakyRelu
from pydtnn.backends.cpu.activations.activation import ActivationCPU

import numpy as np


class LeakyReluCPU(LeakyRelu[np.ndarray], ActivationCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")
        self._mask = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = self._y[:x.shape[0], :]
        self.mask = self._mask[:x.shape[0], :]

        leaky_relu_cython(x.reshape(-1, order="C", copy=False),
                          self.y.reshape(-1, order="C", copy=False),
                          self.mask.reshape(-1, order="C", copy=False),
                          self.negative_slope)
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # return dy * self.mask
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype, order="C")
        return dy
