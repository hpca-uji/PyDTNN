from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.utils.constants import ArrayShape
from pydtnn.libs import numpy as libnp
import numpy as np
import cupy as cp


class LayerCUPY(LayerNumpy):
    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super()._model_init(prev_shape, x)

        if libnp != cp:
            raise RuntimeError("CuPy layers requies PYDTNN_CUPY enabled!")
