"""Feed-forward neural network layer implementation for the PyDTNN framework."""

import logging

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu import Relu
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.fc import FC
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("FeedForward",)

logger = logging.getLogger(__name__)


class FeedForward[T: Array](AbstractBlockLayer[T]):  # noqa: D101 (generics not detected)
    """
    A feed-forward neural network block layer.

    Attributes:
        shape (ArrayShape): The input shape of the layer.
        d_ff (int): The dimensionality of the hidden feed-forward layer.
        dropout_rate (float): The dropout probability applied to the hidden layer.
    """

    def __init__(self, shape: ArrayShape = (), dim_hidden_layers: int = 2048) -> None:
        """Initializes the FeedForward layer. More information: https://arxiv.org/pdf/1706.03762 Section 3.3.

        Args:
            shape (ArrayShape): The input shape of the layer.
            d_ff (int): The dimensionality of the hidden feed-forward layer.
        """
        self.dim_hidden_layers = dim_hidden_layers
        paths: list[list[Layerable[T]]] = [[
            FC(shape=(self.dim_hidden_layers, ), use_bias=True),  # TODO: Check the shape.
            Relu(),
            FC(shape=shape, use_bias=True)
        ]]
        super().__init__(paths)
