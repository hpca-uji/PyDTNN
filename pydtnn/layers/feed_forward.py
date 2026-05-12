"""
Feed-forward neural network layer implementation for the PyDTNN framework.
"""
import logging

from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("FeedForward",)

logger = logging.getLogger(__name__)


class FeedForward[T: Array](AbstractBlockLayer[T]):
    """
    A feed-forward neural network block layer.

    Attributes:
        shape (ArrayShape): The input shape of the layer.
        d_ff (int): The dimensionality of the hidden feed-forward layer.
        dropout_rate (float): The dropout probability applied to the hidden layer.
    """
    def __init__(self, shape: ArrayShape = (1,), d_ff: int = 256, dropout_rate: float = 0.5):
        """
        Initializes the FeedForward layer.

        Args:
            shape (ArrayShape): The input shape of the layer.
            d_ff (int): The dimensionality of the hidden feed-forward layer.
            dropout_rate (float): The dropout probability applied to the hidden layer.
        """
        super().__init__()
        self.shape = shape
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate