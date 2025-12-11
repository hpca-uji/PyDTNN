import numpy as np
from pydtnn.backends.cpu.utils.log_activation_cython import log_bwd_cython, log_fwd_cython

from pydtnn.activations.log import Log
from pydtnn.backends.cpu.activations.activation import ActivationCPU


class LogCPU(Log[np.ndarray], ActivationCPU):

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        # NOTE: These attributes only store data, their value before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self.y = np.zeros(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")
        self.dx = np.zeros(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")

    def _forward_numpy(self, x: np.ndarray) -> np.ndarray:
        # def forward(self, x: np.ndarray) -> np.ndarray:
        y = self.y[:x.shape[0], :]
        # y = np.log(1 / (1 + np.exp(-x)))
        np.multiply(x, -1, out=x,
                    dtype=self.model.dtype)
        np.exp(x, out=x,
               dtype=self.model.dtype)
        np.add(x, 1, out=x,
               dtype=self.model.dtype)
        np.log(x, out=y,
               dtype=self.model.dtype)
        # NOTE: Log propierty: "log(a / b) = log(a) - log(b)", and "log(1) = 0"
        np.multiply(y, -1, out=y,
                    dtype=self.model.dtype)
        return y

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = self.y[:x.shape[0], :]
        log_fwd_cython(x.reshape(-1, copy=False, order="C"), y.reshape(-1, copy=False, order="C"))
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx = self.dx[:dy.shape[0], :]
        log_bwd_cython(dy.reshape(-1, copy=False, order="C"), dx.reshape(-1, copy=False, order="C"))

        return dx
