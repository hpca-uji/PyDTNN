from pydtnn.layers.scalar import Scalar
from pydtnn.backends.numpy.layers.layer import LayerNumpy
from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class ScalarNumpy(Scalar[np.ndarray], LayerNumpy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _model_init(self, prev_shape, x):
        super()._model_init(prev_shape, x)
        # Performance model
        self.fwd_time: np.ndarray = None  # type: ignore # Not yet
        self.bwd_time: np.ndarray = None  # type: ignore # Not yet

    def forward(self, x):
        return x * self.scale

    def backward(self, dy):
        return dy * self.scale
