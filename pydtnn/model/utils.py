"""Utility module for PyDTNN model operations, providing shape and tensor format transformations."""

import logging

import numpy as np

from pydtnn import utils
from pydtnn.model.base import Base
from pydtnn.utils.constants import Array, ArrayShape
from pydtnn.utils.tensor import decode_shape, decode_tensor, encode_shape, encode_tensor

__all__ = ("Utils",)

logger = logging.getLogger(__name__)

DEFAULT_BACH_SIZE = 64
LIMIT_THREADS_AND_BLOCKS = 1024


class Utils[T: Array](Base[T]):  # noqa: D101 (generics not detected)
    """Base utility class for model operations, handling tensor format conversions and configuration access."""

    @property
    def input_shape(self) -> ArrayShape:
        """Returns the shape of the first layer."""
        return self.layers[0].shape

    @property
    def output_shape(self) -> ArrayShape:
        """Returns the shape of the last layer."""
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
        return utils.string_substitute(self.__dict__["dataset_path"], rank=self.comm_rank)

    @property
    def comm_rank(self) -> int:
        """Communicator rank"""
        return self.comm.rank if self.comm else 0

    @property
    def comm_size(self) -> int:
        """Communicator size"""
        return self.comm.size if self.comm else 1

    @property
    def rank(self) -> int:
        """Model process rank"""
        return self.comm_rank if self.shared_data else 0

    @property
    def nprocs(self) -> int:
        """Model process size"""
        return self.comm_size if self.shared_data else 1
