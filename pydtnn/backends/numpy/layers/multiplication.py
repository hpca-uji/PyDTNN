from pydtnn.model import Model
from pydtnn.layers.multiplication import Multiplication
from pydtnn.backends.numpy.layers.layer import LayerNumpy
from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class MultiplicationNumpy(Multiplication[np.ndarray], LayerNumpy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x1 = None
        self.x2 = None

    def _model_init(self, prev_shape, x):
        super()._model_init(prev_shape, x)
        # Performance model
        self.fwd_time = None
        self.bwd_time = None

    def transpose(self, x):
        return x.swapaxes(-2, -1)

    def forward(self, x1, x2):
        if self.model.mode == Model.Mode.TRAIN:
            self.x1 = x1
            self.x2 = x2
        return np.matmul(x1, x2)

    def backward(self, dy):
        dx1 = np.matmul(dy, self.transpose(self.x2))
        dx2 = np.matmul(self.transpose(self.x1), dy)
        return dx1, dx2
