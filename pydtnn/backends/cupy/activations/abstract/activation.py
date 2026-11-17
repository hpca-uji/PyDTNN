"""CuPy-based activation layer implementations for the PyDTNN framework."""

import logging
from typing import TYPE_CHECKING

from pydtnn.libs import numpy as np
from pydtnn.backends.cupy.abstract.layerable import LayerableCupy
from pydtnn.backends.numpy.activations.abstract.activation import ActivationNumpy
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape

__all__ = ("ActivationCupy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class ActivationCupy(ActivationNumpy, LayerableCupy):
    """Base class for activation layers implemented using CuPy for GPU acceleration."""

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """
        Initializes the activation layer, compiling CUDA kernels for forward and backward passes.

        Args:
            prev_shape: The shape of the input tensor.
            x: Optional input data for initialization.
        """
        super()._model_init(prev_shape, x)

        self.defines_replaces = {'"TYPE"': DTYPE2CTYPE[self.model.dtype]}
        self.fwd_kernel = self._fwd_kernel()
        self.bwd_kernel = self._bwd_kernel()
