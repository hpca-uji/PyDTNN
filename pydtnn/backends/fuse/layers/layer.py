"""
Module for handling layer fusion operations in PyDTNN.
"""

from __future__ import annotations

import logging

from pydtnn.layers.abstract.layer import Layer
from pydtnn.utils.constants import Array

__all__ = (
    "LayerFuse",
)

logger = logging.getLogger(__name__)


class LayerFuse[T: Array](Layer):
    """
    Base class for fused layers, allowing initialization from existing layer states.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the fused layer, optionally inheriting state from a parent layer.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments. Supports 'from_parent' to initialize from an existing state.
        """
        from_parent = kwargs.pop("from_parent", None)
        if from_parent is None:
            super().__init__(*args, **kwargs)
        else:
            self.__dict__.update(from_parent)
