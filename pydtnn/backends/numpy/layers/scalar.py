import logging
logger = logging.getLogger(__name__)

from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.layers.scalar import Scalar


class ScalarNumpy(Scalar[np.ndarray], LayerNumpy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _model_init(self, prev_shape, x):
        super()._model_init(prev_shape, x)
        # Performance model
        self.fwd_time = None  # Not yet
        self.bwd_time = self.fwd_time

    def forward(self, x):
        return x * self.scale

    def backward(self, dy):
        return dy * self.scale
