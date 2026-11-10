"""
Encoder module for PyDTNN.

This module provides the Encoder class, which serves as a foundational block
layer for transformer-based architectures within the PyDTNN framework.
"""

import logging

from pydtnn.abstract.layerable import Layerable
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.feed_forward import FeedForward
from pydtnn.layers.layer_normalization import LayerNormalization
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.utils.constants import Array

__all__ = ("Encoder",)

logger = logging.getLogger(__name__)


class Encoder[T: Array](AbstractBlockLayer[T]):  # noqa: D101 (generics not detected)
    """A transformer-based encoder block layer."""

    def __init__(self, embedl: int = 64, dim_keys: int = 3, dim_feed_forward: int = 256, heads: int = 10) -> None:
        """
        Initializes the Encoder layer with specified hyperparameters.

        Args:
            embedl (int): Dimension of the input embeddings.
            dim_keys (int): Dimension of the key/query vectors.
            dim_feed_forward (int): Dimension of the feed-forward network.
            heads (int): Number of attention heads.
        """
        self.embedl = embedl
        self.heads = heads
        self.dim_keys = dim_keys
        self.dim_feed_forward = dim_feed_forward

        # TODO Add Positional Encoding.

        path: list[list[Layerable[T]]] = [
            [
                AdditionBlock([
                    [MultiHeadAttention(heads=self.heads)],
                    []
                ], multiple_inputs=True),
                LayerNormalization(),
                AdditionBlock([
                    [FeedForward(dim_hidden_layers=self.dim_feed_forward)],
                    []
                ]),
                LayerNormalization()
            ]
            # NOTE: One path due is a Sequence.
        ]

        super().__init__(paths=path)

    def backward(self, dy: T) -> T:
        # NOTE: Remove this method after the implementation.
        # Nacho did this at the end of the encoder:
        # dx_3 = self.multiheadattention.backward(dx_3)
        # dx_3, dx_4, dx_5 = dx_3
        # self.dx = dx_2 + dx_3 + dx_4 + dx_5
        # So it's seems necessary to add every output.
        return super().backward(dy)
