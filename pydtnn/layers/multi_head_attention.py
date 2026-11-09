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

    def __init__(self, with_masked_layer: bool = False, heads: int = 8, **kwargs: Any) -> None:
        multi_head_attention_paths = list[Layerable]()
        paths = list[list[Layerable[T]]]()
        for _ in range(heads):
            paths.append(
                [ScaledDotProductAttention(
                    with_masked_layer = with_masked_layer,
                    mult_layers=([[FC(), FC()]], FC()))
                ]
            )
        multi_head_attention_paths.append(ConcatenationBlock[T](paths))
        multi_head_attention_paths.append(FC())

        super().__init__(paths = [multi_head_attention_paths], **kwargs)
