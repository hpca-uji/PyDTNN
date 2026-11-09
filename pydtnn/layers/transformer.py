"""
Tranformer module for PyDTNN.

This module provides the Transformer class, which implements the transformer presented in
"Attention Is All You Need" by Vaswani et al. https://arxiv.org/pdf/1706.03762
"""

import logging

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.encoder import Encoder
from pydtnn.layers.decoder import Decoder
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.fc import FC
from pydtnn.utils.constants import Array

__all__ = ("Encoder",)

logger = logging.getLogger(__name__)


class Transformer[T: Array](AbstractBlockLayer[T]):  # noqa: D101 (generics not detected)
    """A transformer-based encoder block layer."""

    def __init__(self, embedl: int = 64, d_k: int = 3, d_ff: int = 256, heads: int = 10) -> None:
        """
        Initializes the Encoder layer with specified hyperparameters.

        Args:
            embedl (int): Dimension of the input embeddings.
            d_k (int): Dimension of the key/query vectors.
            d_ff (int): Dimension of the feed-forward network.
            heads (int): Number of attention heads.
        """
        self.embedl = embedl
        self.heads = heads
        self.d_k = d_k
        self.d_ff = d_ff

        self.encoder = Encoder()
        self.encoder = Decoder(self.encoder.y)

        # TODO: Add parameters and finish.

        path: list[list[Layerable[T]]] = [
            [
                self.encoder,
                self.encoder,
                FC(),
                Softmax()
            ]
        ]

        super().__init__(paths=path)

    def forward(self, x: T) -> T:
        # TODO: Finish!
        # TODO: Add the positional enconding
        y = x
        # NOTE: Since there is only one path, the code is simplified
        for layer in self.paths[0]:
            y = layer.forward(y)
        # TODO: add the shift (or here or inside the method.)
        self.encoder.set_prev_output(y)
        return y

    def backward(self, dy: T) -> T:
        # TODO: Finish!
        dx = dy
        # NOTE: Since there is only one path, the code is simplified
        for layer in reversed(self.paths[0]):
            dx = layer.backward(dx)
        return dx
