"""Decoder module for the PyDTNN framework."""

import logging
from typing import TYPE_CHECKING

from pydtnn.layers.decoder import Decoder
from pydtnn.backends.numpy.layers.abstract.block_layer import AbstractBlockLayerNumpy
from pydtnn.libs import numpy as np

__all__ = ("Decoder",)

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class DecoderNumpy(Decoder[np.ndarray], AbstractBlockLayerNumpy):
    """Numpy's decoder layer implementation for transformer-based architectures."""
