import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.cupy.abstract.layerable import LayerableCupy
from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.utils.constants import ArrayShape
from pydtnn.libs import numpy as libnp
import numpy as np
import cupy as cp


class LayerCupy(LayerNumpy, LayerableCupy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_compiler = "nvcc"

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super()._model_init(prev_shape, x)

        if libnp != cp:  # type: ignore (It's possible to do this operation)
            raise RuntimeError("CuPy layers requies PYDTNN_CUPY enabled!")
