"""Scalar layer implementation for PyDTNN."""

import logging

from pydtnn.layers.abstract.layer import Layer
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("ProductScalar",)

logger = logging.getLogger(__name__)


class ProductScalar[T: Array](Layer[T]):  # noqa: D101 (generics not detected)
    """A layer that applies a scalar scaling factor to the input."""

    def __init__(self, shape: ArrayShape = (1,), scale: float = 1.0) -> None:
        """
        Initializes the Scalar layer.

        Args:
            shape: The shape of the input data.
            scale: The scalar value to multiply the input by.
        """
        super().__init__(shape)
        self.scale = scale

    def _model_init(self, prev_shape: ArrayShape, x: T | None) -> None:
        super()._model_init(prev_shape, x)
        self.shape = self.prev_shape
