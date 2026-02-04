from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
from pydtnn.activations.relu import Relu
from pydtnn.backends.cpu.activations.activation import ActivationCPU
from pydtnn.utils.constants import ArrayShape


class ReluCPU(Relu[np.ndarray], ActivationCPU):

    def __init__(self, shape: ArrayShape = (1,)):
        super().__init__(shape)
        self.mask: np.ndarray = None  # type: ignore (will be initalized in "initialize")

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        # NOTE: These attributes only store data, their value before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)
        self._mask = np.zeros((self.model.batch_size, *self.prev_shape), dtype=np.int8)

        self.real_memory_size += self._y.nbytes
        self.real_memory_size += self._mask.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = self._y[:x.shape[0], :]
        self.mask = self._mask[:x.shape[0], :]

        np.clip(x, 0, None, out=self.y)
        np.greater(x, 0, out=self.mask)

        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype)
        return dy
