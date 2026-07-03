"""
Encoder module for PyDTNN.

This module provides the Encoder class, which serves as a foundational block
layer for transformer-based architectures within the PyDTNN framework.
"""

import logging

from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.utils.constants import Array

__all__ = ("Encoder",)

logger = logging.getLogger(__name__)


class Encoder[T: Array](AbstractBlockLayer[T]):  # noqa: D101 (generics not detected)
    """
    A transformer-based encoder block layer.

    Attributes:
        embedl (int): Dimension of the input embeddings.
        d_k (int): Dimension of the key/query vectors.
        d_ff (int): Dimension of the feed-forward network.
        heads (int): Number of attention heads.
        dropout_rate (float): Dropout probability.
    """

    def __init__(
        self,
        embedl: int = 64,
        d_k: int = 3,
        d_ff: int = 256,
        heads: int = 10,
        dropout_rate: float = 0.5,
    ) -> None:
        """
        Initializes the Encoder layer with specified hyperparameters.

        Args:
            embedl (int): Dimension of the input embeddings.
            d_k (int): Dimension of the key/query vectors.
            d_ff (int): Dimension of the feed-forward network.
            heads (int): Number of attention heads.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.embedl = embedl
        self.heads = heads
        self.d_k = d_k
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.y = self.dx = None  # type: ignore
