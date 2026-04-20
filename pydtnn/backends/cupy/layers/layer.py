import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.utils.constants import ArrayShape
from pydtnn.libs import numpy as libnp
import cupy as cp

from utils.uses_cuda import CupyCudaCode

class LayerCupy(LayerNumpy, CupyCudaCode):

    BASE_PATH_CODE = f"/backends/cupy/source_modules/"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_compiler = "nvcc"

    def _model_init(self, prev_shape: ArrayShape, x: cp.ndarray | None = None):
        super()._model_init(prev_shape, x)

        if libnp != cp:  # type: ignore (It's possible to do this operation)
            raise RuntimeError("CuPy layers requies PYDTNN_CUPY enabled!")
