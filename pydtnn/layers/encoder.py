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

        # TODO Add Positional Encoding.

        path: list[list[Layerable[T]]] = [
            [
                AdditionBlock([
                    [MultiHeadAttention(heads=self.heads)],
                    []
                ]),
                LayerNormalization(),
                AdditionBlock([
                    [FeedForward(d_ff=self.d_ff)],
                    []
                ]),
                LayerNormalization()
            ]
            # NOTE: One path due is a Sequence.
        ]

        super().__init__(paths=path)
