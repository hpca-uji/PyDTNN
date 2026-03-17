import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.activations.relu import ReluNumpy
from pydtnn.libs import numpy as np

from pydtnn.backends.cython.utils.relu_cython import relu_cython
from pydtnn.backends.cython.activations.activation import ActivationCython


class ReluCython(ReluNumpy, ActivationCython):

    def forward(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        self.y = self._y[:n, :]
        self.mask = self._mask[:n, :]
        relu_cython(x.reshape(-1, copy=False), self.y.reshape(-1, copy=False), self.mask.reshape(-1, copy=False))
        return self.y
