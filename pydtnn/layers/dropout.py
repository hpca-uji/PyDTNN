"""
Dropout layer implementation for PyDTNN.
"""
import logging

from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("Dropout",)

logger = logging.getLogger(__name__)


class Dropout[T: Array](Layer[T]):
    """
    A layer that randomly sets input units to 0 with a frequency of rate during training.
    """
    def __init__(self, rate=0.5):
        """
        Initializes the Dropout layer.

        Args:
            rate: The dropout rate, between 0 and 1.
        """
        super().__init__()
        self.rate = min(1.0, max(0.0, rate))

    def _model_init(self, prev_shape: ArrayShape, x: T | None):
        """
        Initializes layer parameters and shape.

        Args:
            prev_shape: The shape of the input tensor.
            x: Optional input tensor.
        """
        super()._model_init(prev_shape, x)
        self.shape = prev_shape