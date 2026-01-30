from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.layers.input import Input
import numpy as np


class InputCPU(Input[np.ndarray], LayerCPU):

    def _sync_x_y(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_batch = np.asarray(x_batch, dtype=self.model.dtype, order='C', copy=None)
        y_batch = np.asarray(y_batch, dtype=self.model.dtype, order='C', copy=None)
        return x_batch, y_batch

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=self.model.dtype, order="C", copy=None)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return np.asarray(dy, dtype=self.model.dtype, order="C", copy=None)
