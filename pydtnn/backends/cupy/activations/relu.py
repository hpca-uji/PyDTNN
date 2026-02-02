import cupy as np

from pydtnn.activations.relu import Relu
from pydtnn.backends.cupy.activations.activation import ActivationCUPY
from pydtnn.utils.constants import ArrayShape


class ReluCUPY(Relu[np.ndarray], ActivationCUPY):

    def __init__(self, shape: ArrayShape = (1,)):
        super().__init__(shape)
        self.mask: np.ndarray = None  # type: ignore (will be initalized in "initialize")

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        # NOTE: These attributes only store data, their value before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._y = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)
        self._mask = np.zeros((self.model.batch_size, *self.prev_shape), dtype=np.int32)
        self.dx = np.zeros((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = self._y[:x.shape[0], :]
        self.mask = self._mask[:x.shape[0], :]

        np.clip(x, 0, None, out=self.y)
        self.mask[x > 0] = 1
        self.mask[x <= 0] = 0

        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx = self.dx[:dy.shape[0], :]
        np.multiply(dy, self.mask, out=dx, dtype=self.model.dtype)
        return dx
