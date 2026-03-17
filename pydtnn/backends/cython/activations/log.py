import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.activations.log import LogNumpy
from pydtnn.libs import numpy as np
from pydtnn.backends.cython.utils.log_activation_cython import log_bwd_cython, log_fwd_cython

from pydtnn.backends.cython.activations.activation import ActivationCython


class LogCython(LogNumpy, ActivationCython):

    def forward(self, x: np.ndarray) -> np.ndarray:
        y: np.ndarray = self.y[:x.shape[0], :]
        log_fwd_cython(x.reshape(-1, copy=False), y.reshape(-1, copy=False))
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx: np.ndarray = self.dx[:dy.shape[0], :]
        log_bwd_cython(dy.reshape(-1, copy=False), dx.reshape(-1, copy=False))
        return dx
