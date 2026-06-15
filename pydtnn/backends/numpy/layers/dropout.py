"""
Numpy backend implementation of the Dropout layer.
"""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy
from pydtnn.layers.dropout import Dropout
from pydtnn.libs import numpy as np
from pydtnn.model import Model
from pydtnn.utils import random

__all__ = ("DropoutNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class DropoutNumpy(Dropout[np.ndarray], LayerNumpy):
    """
    Numpy-based Dropout layer implementation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the DropoutNumpy layer.
        """
        super().__init__(*args, **kwargs)
        self.mask: np.ndarray = None  # type: ignore (It will be initalized later.)

    def _model_init(self, prev_shape, x=None):
        """
        Initializes layer parameters and calculates memory usage.
        """
        super()._model_init(prev_shape, x)
        self.memory_used += int(math.prod(self.shape)) * self.model.dtype.itemsize

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the dropout operation.

        Args:
            x: Input tensor.

        Returns:
            The input tensor scaled by the dropout mask during training, or unchanged during evaluation.
        """

        match self.model.mode:
            case Model.Mode.TRAIN:
                # NOTE: Remember, it's necessary a new random mask every training's forward call.
                # self.mask = random.binomial(1, (1 - self.rate), size=self.shape).astype(self.model.dtype) / (1 - self.rate)
                self.mask = np.asarray(
                    random.binomial(n=1, p=(1 - self.rate), size=self.shape),
                    dtype=self.model.dtype,
                    order="C",
                )
                np.divide(self.mask, (1 - self.rate), out=self.mask, dtype=self.model.dtype)
                np.multiply(x, self.mask, out=x, dtype=self.model.dtype)
            case Model.Mode.EVALUATE:
                pass  # Just returns x.
            case _:
                raise RuntimeError(f"Unexpected model mode '{self.model.mode}'.")

        return np.asarray(x, dtype=self.model.dtype, order="C")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the dropout operation.

        Args:
            dy: Gradient of the loss with respect to the output.

        Returns:
            The gradient scaled by the dropout mask.
        """
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype)
        return np.asarray(dy, dtype=self.model.dtype, order="C")
