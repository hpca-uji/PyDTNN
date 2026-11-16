"""Scalar layer implementation for PyDTNN."""

import logging
from typing import TYPE_CHECKING

from pydtnn.libs import numpy as np

from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy
from pydtnn.layers.transformer import Transformer

__all__ = ("TransformerNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class TransformerNumpy(Transformer[np.ndarray], LayerNumpy):
    """Numpy's implementation of a Transformer."""
