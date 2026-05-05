import logging

import cupy as cp
import numpy as np

from pydtnn.backends.cupy.abstract.layerable import LayerableCupy
from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.libs import numpy as libnp
from pydtnn.utils.constants import ArrayShape

__all__ = (
    "LayerCupy",
)

logger = logging.getLogger(__name__)


class LayerCupy(LayerNumpy, LayerableCupy):
    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super()._model_init(prev_shape, x)

        if libnp != cp:  # type: ignore (It's possible to do this operation)
            raise RuntimeError("CuPy layers requies PYDTNN_CUPY enabled!")
