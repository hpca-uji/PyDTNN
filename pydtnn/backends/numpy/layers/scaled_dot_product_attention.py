"""Scalar layer implementation for PyDTNN."""

import logging

from pydtnn.layers.scaled_dot_product_attention import ScaledDotProductAttention
from pydtnn.utils.constants import Array

__all__ = ("ScaledDotProductAttentionNumpy",)

logger = logging.getLogger(__name__)


class ScaledDotProductAttentionNumpy[T: Array](ScaledDotProductAttention[T]):  # noqa: D101 (generics not detected)
    """A layer that applies a scalar scaling factor to the input."""
