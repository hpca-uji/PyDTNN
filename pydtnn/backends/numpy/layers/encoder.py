"""Numpy backend implementation of the Transformer Encoder layer."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.block_layer import AbstractBlockLayerNumpy
from pydtnn.layers.encoder import Encoder
from pydtnn.libs import numpy as np

__all__ = ("EncoderNumpy",)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class EncoderNumpy(Encoder[np.ndarray], AbstractBlockLayerNumpy):
    """Numpy-based implementation of the Transformer Encoder layer."""

    def __init__(
            self,
            embedl: int = 64,
            d_k: int = 3,
            d_ff: int = 256,
            heads: int = 10,
        ) -> None:
        """Initializes the EncoderNumpy layer with sub-layers."""
        super().__init__(embedl, d_k, d_ff, heads)
        