from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.layers.input import Input
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class InputNumpy(Input[np.ndarray], LayerNumpy):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=self.model.dtype, order="C")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return np.asarray(dy, dtype=self.model.dtype, order="C")
