"""Multiplication layer module for PyDTNN."""

import logging
from typing import Any

from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.utils.constants import Array

__all__ = ("MultiplicationBlock",)

logger = logging.getLogger(__name__)


class MultiplicationBlock[T: Array](AbstractBlockLayer[T]):  # noqa: D101 (generics not detected)
    """Layer that performs element-wise multiplication of input tensors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the MultiplicationNumpy layer."""
        super().__init__(*args, **kwargs)
        # Following attributes will be initialized later
        self.dx: T = None  # pyright: ignore[reportAttributeAccessIssue]
