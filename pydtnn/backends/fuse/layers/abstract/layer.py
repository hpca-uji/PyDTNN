"""Module for handling layer fusion operations in PyDTNN."""

from __future__ import annotations

import logging
import operator
from functools import reduce
from typing import Any

import numpy as np

from pydtnn.layers.abstract.layer import Layer

__all__ = ("LayerFuse",)

logger = logging.getLogger(__name__)


class LayerFuse(Layer[np.ndarray]):
    """Base class for fused layers, allowing initialization from existing layer states."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the fused layer, optionally inheriting state from a parent layer.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments. Supports 'from_parent' to initialize from an existing state.
        """
        parents = kwargs.pop("parents")

        dict_params = reduce(operator.or_, (layer.__dict__ for layer in reversed(parents)))
        memory_used = reduce(operator.add, (layer.memory_used for layer in reversed(parents)))
        tmp_memory_used = reduce(
            parents[0].model.memory_cls._total,
            (layer.tmp_memory_used for layer in reversed(parents)),
        )
        dict_params |= {
            "parents": parents,
            "memory_used": memory_used,
            "tmp_memory_used": tmp_memory_used,
        }

        self.__dict__.update(dict_params)

    @property
    def canonical_name(self) -> str:
        """Return the class name of the frontend instance."""
        return self.name.removesuffix("Fuse")
