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

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Encoder Numpy backward's implementation."""
        base_dx = super().backward(dy)
        # NOTE: Nacho did this at the end of the encoder:
        # dx_3 = self.multiheadattention.backward(dx_3)
        # dx_3, dx_4, dx_5 = dx_3
        # self.dx = dx_2 + dx_3 + dx_4 + dx_5
        # So it's seems necessary to add every output.

        # NOTE: There are 2 paths at the 1st AdditionLayer, and the MultiHeadAttention backward's output has 3 elements.
        query = base_dx[0][0]
        key = base_dx[0][1]
        value = base_dx[0][2]
        layer_norm = base_dx[1]

        # dx = query + key + value + layer_norm
        np.add(query, key, out=query)
        np.add(query, value, out=query)
        np.add(query, layer_norm, out=query)
        return query
