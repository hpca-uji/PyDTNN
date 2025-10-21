import numpy as np

from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from pydtnn.cython import sigmoid_fwd_cython, sigmoid_bwd_cython


class SigmoidCPU(ActivationCPU, Sigmoid):

    def initialize(self, prev_shape, x = None):
        super().initialize(prev_shape, x)
        self.y: np.ndarray = None

        self._y = np.ndarray(shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype, order="C")
        self.dx = np.ndarray(shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype, order="C")

    def forward(self, x: np.ndarray) -> np.ndarray:

        self.y = self._y[:x.shape[0], :]
        sigmoid_fwd_cython(x.reshape(-1, copy=False), self.y.reshape(-1, copy=False))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:

        dx = self.dx[:dy.shape[0], :]
        sigmoid_bwd_cython(dy.reshape(-1, copy=False, order="C"), 
                           self.y.reshape(-1, copy=False, order="C"), 
                           dx.reshape(-1, copy=False, order="C"))

        return dx
