import logging
from typing import Any

import numpy as np

from pydtnn import utils
from pydtnn.model_parts.base import Base
from pydtnn.utils.constants import Array, ArrayShape
from pydtnn.utils.tensor import (decode_shape, decode_tensor, encode_shape,
                                 encode_tensor)

logger = logging.getLogger(__name__)

BAR_WIDTH = 150
DEFAULT_BACH_SIZE = 64
LIMIT_THREADS_AND_BLOCKS = 1024


class Utils[T: Array](Base[T]):

    @property
    def input_shape(self):
        return self.layers[0].shape

    @property
    def output_shape(self):
        return self.layers[-1].shape

    def encode_shape(self, shape: ArrayShape) -> ArrayShape:
        """Transform the shape from `NCHW` order to `model.tensor_format` order (supports 4 or 3 dimensions)"""
        return encode_shape(shape, self.tensor_format)

    def decode_shape(self, shape: ArrayShape) -> ArrayShape:
        """Transform the shape from `model.tensor_format` order to `NCHW` order (supports 4 or 3 dimensions)."""
        return decode_shape(shape, self.tensor_format)

    def encode_tensor(self, data: np.ndarray) -> np.ndarray:
        """Transpose elements of data from `NCHW` format to `model.tensor_format` format (supports 4 or 3 dimensions)."""
        return encode_tensor(data, self.tensor_format)

    def decode_tensor(self, data: np.ndarray) -> np.ndarray:
        """Transpose elements of data from `model.tensor_format` format to `NCHW` format (supports 4 or 3 dimensions)."""
        return decode_tensor(data, self.tensor_format)

    @property
    def dataset_path(self) -> str:
        """Raw dataset path with rank substituted"""
        return utils.string_substitute(self.kwargs["dataset_path"], rank=self.comm_rank)

    def __getattr__(self, item) -> Any:
        return self.kwargs.get(item)
