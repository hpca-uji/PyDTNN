import logging

from pydtnn.backends.cython.activations.activation import ActivationCython
from pydtnn.backends.cython.utils.relu_cython import leaky_relu_cython
from pydtnn.backends.numpy.activations.leaky_relu import LeakyReluNumpy
from pydtnn.libs import numpy as np

__all__ = (
    "LeakyReluCython",
)

logger = logging.getLogger(__name__)


class LeakyReluCython(LeakyReluNumpy, ActivationCython):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = self._y[:x.shape[0], :]
        self.mask = self._mask[:x.shape[0], :]

        leaky_relu_cython(x.reshape(-1, copy=False),
                          self.y.reshape(-1, copy=False),
                          self.mask.reshape(-1, copy=False),
                          self.negative_slope)
        return self.y
