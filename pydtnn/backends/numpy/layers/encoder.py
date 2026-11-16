"""Numpy backend implementation of the Transformer Encoder layer."""

import logging
from typing import TYPE_CHECKING

from numpy import ndarray

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
        dim_keys: int = 3,
        dim_feed_forward: int = 256,
        heads: int = 10,
    ) -> None:
        """Initializes the EncoderNumpy layer with sub-layers."""
        super().__init__(embedl, dim_keys, dim_feed_forward, heads)

    def backward(self, dy: ndarray) -> ndarray:
        """Encoder Numpy backward's implementation."""
        base_dx = super().backward(dy)
        # NOTE: Remove this method after the implementation.
        # Nacho did this at the end of the encoder:
        # dx_3 = self.multiheadattention.backward(dx_3)
        # dx_3, dx_4, dx_5 = dx_3
        # self.dx = dx_2 + dx_3 + dx_4 + dx_5
        # So it's seems necessary to add every output.

        # NOTE: Hay 2 paths en el 1er AdditionLayer, y la salida del backwards MultiHeadAttention tiene 3 elementos.
        query = base_dx[0][0]
        key = base_dx[0][1]
        value = base_dx[0][2]
        layer_norm = base_dx[1]

        # dx = query + key + value + layer_norm
        np.add(query, key, out=query)
        np.add(query, value, out=query)
        np.add(query, layer_norm, out=query)
        return query
