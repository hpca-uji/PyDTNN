from pydtnn.backends.cpu.activations.sigmoid import SigmoidCPU
from pydtnn.libs import numpy as np
from pydtnn.backends.cython.utils.sigmoid_cython import sigmoid_bwd_cython, sigmoid_fwd_cython

from pydtnn.backends.cython.activations.activation import ActivationCYTHON


class SigmoidCYTHON(SigmoidCPU, ActivationCYTHON):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y: np.ndarray = self._y[:x.shape[0], :]
        sigmoid_fwd_cython(x.reshape(-1, copy=False), self.y.reshape(-1, copy=False))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx: np.ndarray = self.dx[:dy.shape[0], :]
        sigmoid_bwd_cython(dy.reshape(-1, copy=False),
                           self.y.reshape(-1, copy=False),
                           dx.reshape(-1, copy=False))
        return dx
