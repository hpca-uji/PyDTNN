"""2D Max Pooling layer implementation for the PyDTNN framework."""

import logging

from pydtnn.layers.abstract.layer import Layer
from pydtnn.utils.constants import Array

__all__ = ("Log",)

logger = logging.getLogger(__name__)


class Log[T: Array](Layer[T]):  # noqa: D101 (generics not detected)
    """Performs log on the input tensor."""
