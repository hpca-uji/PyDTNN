import numpy as np
from pydtnn.backends.cpu.utils.relu_cython import capped_relu_cython
from pydtnn.activations.relu6 import Relu6
from pydtnn.backends.cpu.activations.activation import ActivationCPU


class Relu6CPU(Relu6[np.ndarray], ActivationCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask: np.ndarray = None # type: ignore (will be initalized in "initialize")

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        # NOTE: These attributes only store data, their value before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")
        self._mask = np.zeros((self.model.batch_size, *self.prev_shape), dtype=np.int8, order="C")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y: np.ndarray = self._y[:x.shape[0], :]
        self.mask: np.ndarray = self._mask[:x.shape[0], :]
        capped_relu_cython(x.reshape(-1, copy=False, order="C"),
                           self.y.reshape(-1, copy=False, order="C"),
                           self.mask.reshape(-1, copy=False, order="C"),
                           self.cap)
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # return dy * self.mask
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype, order="C")
        return dy
