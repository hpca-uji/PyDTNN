import logging

import numpy as np
from cupy.cuda import Stream  # type: ignore

from pydtnn.backends.cupy.layers.layer import LayerCupy
from pydtnn.backends.numpy.layers.fc import FCNumpy
from pydtnn.utils.constants import ArrayShape

__all__ = (
    "FCCupy",
)

logger = logging.getLogger(__name__)


class FCCupy(FCNumpy, LayerCupy):

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)

        self.stream_2 = Stream()
