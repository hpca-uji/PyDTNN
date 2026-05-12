"""
ReLU6 activation module for PyDTNN.
"""
import logging

from pydtnn.activations.relu import Relu
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("Relu6",)

logger = logging.getLogger(__name__)


# NOTE -> "CappedRelu": https://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf


class Relu6[T: Array](Relu[T]):
    """
    Capped ReLU activation layer that limits the output to a specified maximum value.
    """
    # NOTE: This is a ReLU6 *iif* cap is 6, but it's more interesting a implementation where the user have the freedom to choose their cap.
    def __init__(self, shape: ArrayShape = (1,), cap: float = 6.0):
        """
        Initializes the Relu6 layer.

        Args:
            shape: The shape of the input tensor.
            cap: The upper bound for the activation output.
        """
        super().__init__(shape)
        self.cap: float = cap