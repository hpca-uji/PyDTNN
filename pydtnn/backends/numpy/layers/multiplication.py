"""
Numpy backend implementation of the Multiplication layer.
"""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy
from pydtnn.layers.multiplication import Multiplication
from pydtnn.libs import numpy as np
from pydtnn.model import Model

__all__ = ("MultiplicationNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class MultiplicationNumpy(Multiplication[np.ndarray], LayerNumpy):
    """
    Numpy-based multiplication layer for matrix operations.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the MultiplicationNumpy layer.
        """
        super().__init__(*args, **kwargs)
        self.x1 = None
        self.x2 = None

    def _model_init(self, prev_shape, x):
        """
        Initialize model-specific parameters and performance tracking.
        """
        super()._model_init(prev_shape, x)
        # Performance model
        self.fwd_time = None  # type: ignore (defined later)
        self.bwd_time = None  # type: ignore (defined later)

    def transpose(self, x):
        """
        Transpose the last two dimensions of the input array.
        """
        return x.swapaxes(-2, -1)

    def forward(self, x1, x2):
        """
        Perform matrix multiplication of two inputs.
        """
        if self.model.mode == Model.Mode.TRAIN:
            self.x1 = x1
            self.x2 = x2
        return np.matmul(x1, x2)

    def backward(self, dy):
        """
        Compute gradients with respect to the inputs.
        """
        dx1 = np.matmul(dy, self.transpose(self.x2))
        dx2 = np.matmul(self.transpose(self.x1), dy)
        return dx1, dx2
