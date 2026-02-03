from pydtnn.libs import numpy as np
from pydtnn.backends.cython.utils.relu_cython import capped_relu_cython
from pydtnn.backends.cpu.activations.relu6 import Relu6CPU
from pydtnn.backends.cython.activations.activation import ActivationCYTHON


class Relu6CYTHON(Relu6CPU, ActivationCYTHON):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y: np.ndarray = self._y[:x.shape[0], :]
        self.mask: np.ndarray = self._mask[:x.shape[0], :]
        capped_relu_cython(x.reshape(-1, copy=False),
                           self.y.reshape(-1, copy=False),
                           self.mask.reshape(-1, copy=False),
                           self.cap)
        return self.y
