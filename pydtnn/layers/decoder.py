"""
Decoder module for the PyDTNN framework.
"""
import logging

from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.utils.constants import Array

__all__ = ("Decoder",)

logger = logging.getLogger(__name__)


class Decoder[T: Array](AbstractBlockLayer[T]):
    """
    Decoder layer implementation for transformer-based architectures.
    """
    def __init__(self, embedl: int = 64, d_k: int = 3, d_ff: int = 256, heads: int = 10, dropout_rate: float = 0.5):
        """
        Initializes the Decoder layer.

        Args:
            embedl: Embedding dimension size.
            d_k: Dimension of the key/query/value vectors.
            d_ff: Dimension of the feed-forward network.
            heads: Number of attention heads.
            dropout_rate: Dropout probability.
        """
        super().__init__()
        self.embedl = embedl
        self.heads = heads
        self.d_k = d_k
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate