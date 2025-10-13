import numpy as np

from pydtnn.activations.log import Log
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from numpy import ndarray
from pydtnn.cython_modules import log_fwd_cython, log_bwd_cython


class LogCPU(ActivationCPU, Log):

    def initialize(self, prev_shape, x = None):
        super().initialize(prev_shape, x)
        self.y = np.empty(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")
        self.dx = np.empty(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")

    def forward(self, x: ndarray) -> ndarray:
        y = self.y[:x.shape[0], :]
        log_fwd_cython(x.reshape(-1, copy=False, order="C"), y.reshape(-1, copy=False, order="C"))
        return y

    def backward(self, dy: ndarray) -> ndarray:
        dx = self.dx[:dy.shape[0], :]
        log_bwd_cython(dy.reshape(-1, copy=False, order="C"), dx.reshape(-1, copy=False, order="C"))

        return dx
