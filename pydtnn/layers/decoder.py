"""Decoder module for the PyDTNN framework."""

import logging

from pydtnn.abstract.layerable import Layerable
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.feed_forward import FeedForward
from pydtnn.layers.identity import Identity
from pydtnn.layers.layer_normalization import LayerNormalization
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.utils.constants import Array

__all__ = ("Decoder",)

logger = logging.getLogger(__name__)


class Decoder[T: Array](AbstractBlockLayer[T]):  # noqa: D101 (generics not detected)
    """Decoder layer implementation for transformer-based architectures."""

    def __init__(self,
                 encoder_y: T,
                 embedl: int = 64,
                 d_k: int = 3,
                 d_ff: int = 256,
                 heads: int = 10,
        ) -> None:
        """
        Initializes the Encoder layer with specified hyperparameters.

        Args:
            embedl (int): Dimension of the input embeddings.
            d_k (int): Dimension of the key/query vectors.
            d_ff (int): Dimension of the feed-forward network.
            heads (int): Number of attention heads.
        """
        self.encoder_y: T = encoder_y
        self.embedl: int = embedl
        self.heads: int = heads
        self.d_k: int = d_k
        self.d_ff: int = d_ff
        # TODO Add Positional Encoding and output shifted right.

        # It will be initalized later
        self.prev_output: T = None  # pyright: ignore[reportAttributeAccessIssue]
         
        # NOTE: Va a tocar hacer algo un poco raro: Crear una capa que ignore la entraad devuelva el Tensor que le has seteado.
        self.special_layer = Identity()

        # NOTE: The layer is divided in 2 paths due it's special nature:
        #   - The input from the "previous layer" it goes directly to the second path (the encoder).
        #   - The output from the previous iteretion (shifted to the right) is the first path input.
        path: list[list[Layerable[T]]] = [
            [
                AdditionBlock([
                    [MultiHeadAttention(with_masked_layer=True,
                                        heads=self.heads)],
                    []
                ]),
                LayerNormalization(),
            ],
            [
                AdditionBlock([
                    [MultiHeadAttention(heads=self.heads)],
                    [self.special_layer]
                ]),
                LayerNormalization(),
                AdditionBlock([
                    [FeedForward(d_ff=self.d_ff)],
                    []
                ]),
                LayerNormalization()

            ]
        ]

        super().__init__(paths=path)

    def set_prev_output(self, prev_output: T) -> None:
        """Set the value of the previous output"""
        # TODO: add the shift (in that case, change the name to reflec this operation.)
        self.prev_output = prev_output

    def forward(self, x: T) -> T:
        multihead_section = self.paths[0]
        encoder_section = self.paths[1]

        if self.prev_output is not None:
            prev_y = self.prev_output
            for layer in multihead_section:
                prev_y = layer.forward(prev_y)

        # TODO:
        self.special_layer.set_return(prev_y)

        y = x
        for layer in encoder_section:
            y = layer.forward(y)
        return y  # pyright: ignore[reportPossiblyUnboundVariable]

    def backward(self, dy: T) -> T:
        for path in self.paths:
            dx = dy
            for layer in reversed(path):
                dx = layer.backward(dx)
        return dx  # pyright: ignore[reportPossiblyUnboundVariable]