import numpy as np

from pydtnn.activations.arctanh import Arctanh
from pydtnn.backends.cpu.activations.activation import ActivationCPU


class ArctanhCPU(Arctanh[np.ndarray], ActivationCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        # NOTE: This attribute only stores data, its value before the operation doesn't matters; it's initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros(shape=(self.model.batch_size, *self.shape),
                           dtype=self.model.dtype, order="C")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = self._y[:x.shape[0], :]
        np.arctanh(x, out=self.y, casting="unsafe", dtype=self.model.dtype, order="C")
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # return 1 / (1 + dy ** 2)
        np.power(dy, 2, out=dy,
                 casting="unsafe", dtype=self.model.dtype)
        np.add(dy, 1, out=dy,
               casting="unsafe", dtype=self.model.dtype)
        np.reciprocal(dy, out=dy, casting="unsafe", dtype=self.model.dtype, order="C")
        return dy
