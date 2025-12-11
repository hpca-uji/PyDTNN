import numpy as np

from pydtnn.backends.cpu.utils.relu_cython import relu_cython
from pydtnn.activations.relu import Relu
from pydtnn.backends.cpu.activations.activation import ActivationCPU
from pydtnn.utils.constants import ArrayShape


class ReluCPU(Relu[np.ndarray], ActivationCPU):

    def __init__(self, shape: ArrayShape = (1,)):
        super().__init__(shape)
        self.mask: np.ndarray = None # type: ignore (will be initalized in "initialize")

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        # NOTE: These attributes only store data, their value before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")
        self._mask = np.zeros((self.model.batch_size, *self.prev_shape), dtype=np.int8, order="C")
        self.dx = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")

    def forward(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        self.y = self._y[:n, :]
        self.mask = self._mask[:n, :]
        relu_cython(x.reshape(-1, copy=False, order="C"), self.y.reshape(-1, copy=False, order="C"), self.mask.reshape(-1, copy=False, order="C"))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx = self.dx[:dy.shape[0], :]
        np.multiply(dy, self.mask, out=dx, dtype=self.model.dtype, order="C")
        return dx
