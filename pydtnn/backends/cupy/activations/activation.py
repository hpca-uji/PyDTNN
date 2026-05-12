"""
CuPy-based activation layer implementations for the PyDTNN framework.
"""
import logging

import cupy as cp
import numpy as np

from pydtnn.backends.cupy.abstract.layerable import LayerableCupy
from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.libs import numpy as libnp
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape

__all__ = ("ActivationCupy",)

logger = logging.getLogger(__name__)


class ActivationCupy(ActivationNumpy, LayerableCupy):
    """
    Base class for activation layers implemented using CuPy for GPU acceleration.
    """
    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        """
        Initializes the activation layer, compiling CUDA kernels for forward and backward passes.

        Args:
            prev_shape: The shape of the input tensor.
            x: Optional input data for initialization.
        """
        super()._model_init(prev_shape, x)

        if libnp != cp:  # type: ignore (It's possible to do this operation)
            raise RuntimeError("CuPy layers requies PYDTNN_CUPY enabled!")

        self.defines_replaces = {'"TYPE"': DTYPE2CTYPE[self.model.dtype]}
        self.fwd_kernel = self._fwd_kernel()
        self.bwd_kernel = self._bwd_kernel()