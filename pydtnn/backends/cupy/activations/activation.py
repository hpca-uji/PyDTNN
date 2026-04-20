from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.backends.cupy.abstract.layerable import LayerableCupy
import logging

from pydtnn.utils.constants import ArrayShape
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.utils.constants import DTYPE2CTYPE

from pydtnn.libs import numpy as libnp
import cupy as cp

from utils.uses_cuda import CupyCudaCode

class ActivationCupy(ActivationNumpy, LayerableCupy, CupyCudaCode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_compiler = "nvcc"

    def _model_init(self, prev_shape: ArrayShape, x: cp.ndarray | None = None):
        super()._model_init(prev_shape, x)

        if libnp != cp:  # type: ignore (It's possible to do this operation)
            raise RuntimeError("CuPy layers requies PYDTNN_CUPY enabled!")

        self.defines_replaces = {"\"TYPE\"": DTYPE2CTYPE[self.model.dtype]}
        self.fwd_kernel = self._fwd_kernel()
        self.bwd_kernel = self._bwd_kernel()
