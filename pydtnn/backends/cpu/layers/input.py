from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.layers.input import Input
import numpy as np


class InputCPU(LayerCPU, Input[np.ndarray]):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=self.model.dtype, order="C", copy=None)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return np.asarray(dy, dtype=self.model.dtype, order="C", copy=None)
