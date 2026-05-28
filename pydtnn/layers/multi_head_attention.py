"""
Multi-head attention layer implementation for PyDTNN.
"""

import logging

from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.utils import initializers
from pydtnn.utils.constants import Array

__all__ = ("MultiHeadAttention",)

logger = logging.getLogger(__name__)


class MultiHeadAttention[T: Array](AbstractBlockLayer[T]):
    """
    Multi-head attention mechanism layer.
    """

    def __init__(
        self,
        embedl: int = 64,
        d_k: int = 3,
        heads: int = 10,
        dropout_rate: float = 0.5,
        weights_initializer=initializers.glorot_uniform,
        biases_initializer=initializers.zeros,
    ):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            embedl: Embedding dimension size.
            d_k: Dimension of each attention head.
            heads: Number of attention heads.
            dropout_rate: Dropout probability.
            weights_initializer: Initializer for weight matrices.
            biases_initializer: Initializer for bias vectors.
        """
        super().__init__()
        self.embedl = embedl
        self.heads = heads
        self.d_k = d_k
        self.dropout_rate = dropout_rate
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer
