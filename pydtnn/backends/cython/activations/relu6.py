import logging

from pydtnn.backends.cython.activations.activation import ActivationCython
from pydtnn.backends.cython.utils.relu_cython import capped_relu_cython
from pydtnn.backends.numpy.activations.relu6 import Relu6Numpy
from pydtnn.libs import numpy as np

logger = logging.getLogger(__name__)


class Relu6Cython(Relu6Numpy, ActivationCython):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y: np.ndarray = self._y[:x.shape[0], :]
        self.mask: np.ndarray = self._mask[:x.shape[0], :]
        capped_relu_cython(x.reshape(-1, copy=False),
                           self.y.reshape(-1, copy=False),
                           self.mask.reshape(-1, copy=False),
                           self.cap)
        return self.y
