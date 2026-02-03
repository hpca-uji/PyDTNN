from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.utils.constants import ArrayShape
from pydtnn.libs import numpy as libnp
import numpy as np
import cupy as cp


class LayerCUPY(LayerCPU):
    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)

        if libnp != cp:
            raise RuntimeError("CuPy layers requies PYDTNN_CUPY enabled!")
