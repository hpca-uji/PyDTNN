"""Decoder module for the PyDTNN framework."""

import logging

from pydtnn.abstract.layerable import Layerable
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.feed_forward import FeedForward
from pydtnn.layers.forwarded_output import ForwardedOutput
from pydtnn.layers.layer_normalization import LayerNormalization
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.utils.constants import Array

__all__ = ("Decoder",)

logger = logging.getLogger(__name__)


class Decoder[T: Array](AbstractBlockLayer[T]):  # noqa: D101 (generics not detected)
    """Decoder layer implementation for transformer-based architectures. More info: https://arxiv.org/pdf/1706.03762"""

    def __init__(
        self,
        encoder_y: T,
        embedl: int = 64,
        dim_keys: int = 3,
        dim_feed_forward: int = 256,
        heads: int = 10,
    ) -> None:
        """
        Initializes the Encoder layer with specified hyperparameters.

        Args:
            embedl (int): Dimension of the input embeddings.
            dim_keys (int): Dimension of the key/query vectors.
            dim_feed_forward (int): Dimension of the feed-forward network.
            heads (int): Number of attention heads.
        """
        self.encoder_y: T = encoder_y
        self.embedl: int = embedl
        self.heads: int = heads
        self.dim_keys: int = dim_keys
        self.dim_feed_forward: int = dim_feed_forward
        # TODO Add Positional Encoding and output shifted right.
        self.masked_mha_output = ForwardedOutput(use_fwd_seted_value=True)

        # It will be initalized later
        self.prev_output: T = None  # pyright: ignore[reportAttributeAccessIssue]
        self.masked_multi_head_dx: T = None  # pyright: ignore[reportAttributeAccessIssue]

        # NOTE: The layer is divided in 2 paths due it's special nature:
        #   - The input from the "previous layer" it goes directly to the second path (the encoder).
        #   - The output from the previous iteretion (shifted to the right) is the first path input.
        path: list[list[Layerable[T]]] = [
            [
                # Masked multi-head attention block
                AdditionBlock([
                    [MultiHeadAttention(with_masked_layer=True,
                                        heads=self.heads)],
                    []
                ]),
                LayerNormalization(),
            ],
            [
                # Internal encoder Block
                AdditionBlock([
                    [MultiHeadAttention(heads=self.heads)],
                    [self.masked_mha_output]
                ]),
                LayerNormalization(),
                AdditionBlock([
                    [FeedForward(dim_hidden_layers=self.dim_feed_forward)],
                    []
                ]),
                LayerNormalization()
            ]
        ]

        super().__init__(paths=path)

    def set_prev_output(self, prev_output: T) -> None:
        """Set the value of the previous output"""
        self.prev_output = prev_output

    def forward(self, x: T) -> T:
        multihead_section = self.paths[0]
        encoder_section = self.paths[1]

        # The Masked Multi-Head Attention and 1st Add & Norm section:
        if self.prev_output is not None:
            prev_y = self.prev_output
            for layer in multihead_section:
                prev_y = layer.forward(prev_y)
            # NOTE: The following line is for setting the input of the next
            #    Multihead Attention layer and the "Add & Norm" that comes after.
            self.masked_mha_output.set_forward_output(prev_y)

        # From the 2nd Multi-Head Attention section to the end.
        y = x
        for layer in encoder_section:
            y = layer.forward(y)
        return y  # pyright: ignore[reportPossiblyUnboundVariable]

    def backward(self, dy: T) -> T:
        # From the end to the 2nd Multi-Head Attention.
        dx_encoder = dy
        for layer in reversed(self.paths[1]):
            dx_encoder = layer.backward(dx_encoder)

        # TODO: Check how to divide and operate with the output.
        multiheadattention_out: T = dx_encoder[0]  # pyright: ignore[reportAssignmentType]
        layernormalization_out: T = dx_encoder[1]  # pyright: ignore[reportAssignmentType]

        # From the 2nd Multi-Head Attention to the beginning.
        masked_multi_head_dx = layernormalization_out
        for layer in reversed(self.paths[0]):
            masked_multi_head_dx = layer.backward(masked_multi_head_dx)

        # TODO: Check how to divide and operate with the output (again).
        self.masked_multi_head_dx = masked_multi_head_dx
        dx = multiheadattention_out

        return dx
