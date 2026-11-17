"""Scalar layer implementation for PyDTNN."""

import logging
from typing import Any

from pydtnn.abstract.layerable import Layerable
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.fc import FC
from pydtnn.layers.scaled_dot_product_attention import ScaledDotProductAttention
from pydtnn.utils.constants import Array

__all__ = ("MultiHeadAttention",)

logger = logging.getLogger(__name__)


class MultiHeadAttention[T: Array](AbstractBlockLayer[T]):  # noqa: D101 (generics not detected)
    """Implementation of a Multi-Head Attention. https://arxiv.org/pdf/1706.03762, Figure 2 (right)."""

    def __init__(self, scale_factor: float, with_masked_layer: bool = False, heads: int = 8, **kwargs: Any) -> None:
        self.heads = heads
        multi_head_attention_paths = list[Layerable[T]]()
        paths = list[list[Layerable[T]]]()
        for _ in range(self.heads):
            paths.append(
                [ScaledDotProductAttention(
                    scale_factor=scale_factor,
                    with_masked_layer=with_masked_layer,
                    mult_layers=([[FC(), FC()]], FC())
                )]
            )
        multi_head_attention_paths.append(ConcatenationBlock[T](paths, multiple_inputs=True))
        multi_head_attention_paths.append(FC())

        super().__init__(paths=[multi_head_attention_paths], **kwargs)

    def _split(self, x: T) -> T:
        raise NotImplementedError(f"_split method not implemented in \'{self.backend_name}\' backend.")

    def _concat(self, dx: T) -> T:
        raise NotImplementedError(f"_concat method not implemented in \'{self.backend_name}\' backend.")

    def forward(self, x: T) -> T:
        """Forward method."""
        x = self._split(x)
        # NOTE: Remember: Input for every concatenation path must be divided by the number of heads!!
        return super().forward(x)

    def backward(self, dy: T) -> T:
        """Forward method."""
        # NOTE: CHECK THIS.
        # NOTE: Remember: Outupt for every "concatenation path" must be concatenated in order to regain it's dimensionality.
        dx = super().forward(dy)
        return self._concat(dx)
