"""
Dataset module for PyDTNN.

Provides the base Dataset class and utility functions for managing,
transforming, and generating data batches for machine learning models.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Generator

import numpy as np

from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import SampleFormat, TensorFormat, format_transpose

__all__ = ("Base",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model.utils import Utils as Model


type TransformFunc = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class Base:
    """
    Abstract base class for all datasets in PyDTNN.

    Defines the interface and common utilities for data partitioning,
    shape management, and format conversion.
    """

    class Part(IntEnum):
        """Enum representing the dataset partition."""

        TRAIN = 0
        VAL = 1
        TEST = 2

    model: Model
    debug: bool
    test_as_validation: bool
    x_empty_batch: np.ndarray[tuple[int, ...]]
    y_empty_batch: np.ndarray[tuple[int, ...]]
    input_shape: ArrayShape
    output_shape: ArrayShape
    _nsamples: list[int]
    _transformations: dict[Base.Part, list[TransformFunc]]
    _initial_nsamples: list[int]
    _local_offset: list[int]
    _local_nsamples: list[int]
    _local_remaining_nsamples: list[int]
    _x: list[np.ndarray[tuple[int, ...]]]
    _y: list[np.ndarray[tuple[int, ...]]]
    _data_generator: Callable[[Base.Part], Generator[tuple[np.ndarray, np.ndarray]]]

    @property
    def name(self) -> str:
        """Return the class name of the dataset."""
        return type(self).__name__

    @property
    def train_nsamples(self):
        """Get number of training samples."""
        return self._nsamples[Base.Part.TRAIN]

    @train_nsamples.setter
    def train_nsamples(self, value):
        """Set number of training samples."""
        self._nsamples[Base.Part.TRAIN] = value

    @property
    def val_nsamples(self):
        """Get number of validation samples."""
        return self._nsamples[Base.Part.VAL]

    @val_nsamples.setter
    def val_nsamples(self, value):
        """Set number of validation samples."""
        self._nsamples[Base.Part.VAL] = value

    @property
    def test_nsamples(self):
        """Get number of test samples."""
        return self._nsamples[Base.Part.TEST]

    @test_nsamples.setter
    def test_nsamples(self, value):
        """Set number of test samples."""
        self._nsamples[Base.Part.TEST] = value

    @staticmethod
    def _nchw2nhwc(x: np.ndarray) -> np.ndarray:
        """Convert NCHW tensor to NHWC."""
        return format_transpose(x, TensorFormat.NCHW, TensorFormat.NHWC)

    @staticmethod
    def _nhwc2nchw(x: np.ndarray) -> np.ndarray:
        """Convert NHWC tensor to NCHW."""
        return format_transpose(x, TensorFormat.NHWC, TensorFormat.NCHW)

    @staticmethod
    def _chw2hwc(x: np.ndarray) -> np.ndarray:
        """Convert CHW sample to HWC."""
        return format_transpose(x, SampleFormat.CHW, SampleFormat.HWC)

    @staticmethod
    def _hwc2chw(x: np.ndarray) -> np.ndarray:
        """Convert HWC sample to CHW."""
        return format_transpose(x, SampleFormat.HWC, SampleFormat.CHW)

    @staticmethod
    def _decode_class(y: np.ndarray, classes_list: np.ndarray) -> None:
        """Sets to 1 the corresponding entry in the 2D y array as indicated by the 1D array of classes"""
        y[np.arange(y.shape[0]), classes_list] = 1

    @staticmethod
    def _offset2files(filenames: list[str], images_per_file: int, local_offset: int, local_nsamples: int) -> list[tuple[str, int, int]]:
        """
        Map local offset and sample count to specific files.

        Given a list of filenames, the number of samples per file, a local offset,
        and a local sample count, this method determines which files and which
        ranges within those files contain the required samples.

        Args:
            filenames: A list of filenames.
            images_per_file: The number of samples contained in each file.
            local_offset: The starting sample index for the current request.
            local_nsamples: The total number of samples to retrieve.

        Returns:
            A list of tuples, where each tuple contains:
            (filename, offset_in_file, num_samples_from_file).
        """
        i = local_offset // images_per_file
        offset_in_file = local_offset - i * images_per_file
        output = []
        while local_nsamples:
            nsamples = min(images_per_file - offset_in_file, local_nsamples)
            output.append((filenames[i], offset_in_file, nsamples))
            offset_in_file = 0
            local_nsamples -= nsamples
        return output
