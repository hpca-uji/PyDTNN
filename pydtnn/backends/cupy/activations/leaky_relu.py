import logging

from pydtnn.backends.cupy.activations.activation import ActivationCupy
from pydtnn.backends.numpy.activations.leaky_relu import LeakyReluNumpy
from pydtnn.libs import numpy as np

logger = logging.getLogger(__name__)


class LeakyReluCupy(LeakyReluNumpy, ActivationCupy):

    def _model_init(self, prev_shape, x=None):
        super()._model_init(prev_shape, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = np.ascontiguousarray(self._y[:x.shape[0], :], dtype=self.model.dtype)
        self.mask = np.ascontiguousarray(self._mask[:x.shape[0], :], dtype=self.model.dtype)

        self.fwd_kernel(self.model.cuda_grid,
                        self.model.cuda_block,
                        (x, self.y, self.mask, self.negative_slope, x.size))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.bwd_kernel(self.model.cuda_grid,
                        self.model.cuda_block,
                        (dy, dy, self.mask, self.negative_slope, dy.size))
        return dy
