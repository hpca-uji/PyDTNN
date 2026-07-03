"""CuPy backend implementation for neural network layers."""

import logging

import cupy as cp
import numpy as np

from pydtnn.backends.cupy.abstract.layerable import LayerableCupy
from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy
from pydtnn.libs import numpy as libnp
from pydtnn.utils.constants import ArrayShape

__all__ = ("LayerCupy",)

logger = logging.getLogger(__name__)


class LayerCupy(LayerNumpy, LayerableCupy):
    """Base class for layers using the CuPy backend."""

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """
        Initialize the layer model parameters and verify backend compatibility.

        Args:
            prev_shape: The shape of the input data from the previous layer.
            x: Optional input data for initialization.
        """
        super()._model_init(prev_shape, x)

        if libnp != cp:  # type: ignore (It's possible to do this operation)
            raise RuntimeError("CuPy layers requies PYDTNN_CUPY enabled!")
