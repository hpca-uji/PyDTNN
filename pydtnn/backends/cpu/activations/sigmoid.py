import numpy as np
from pydtnn.backends.cpu.utils.sigmoid_cython import sigmoid_bwd_cython, sigmoid_fwd_cython

from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.backends.cpu.activations.activation import ActivationCPU


class SigmoidCPU(Sigmoid[np.ndarray], ActivationCPU):

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        self.y: np.ndarray = None  # type: ignore (the value will be set in forward)

        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros(shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype, order="C")
        self.dx = np.zeros(shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype, order="C")

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
