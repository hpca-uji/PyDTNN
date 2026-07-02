"""
Dataset module for PyDTNN.

Provides the base Dataset class and utility functions for managing,
transforming, and generating data batches for machine learning models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import IO, Callable

import numpy as np
import rapidgzip
from PIL import Image

from pydtnn.datasets.abstract.base import Base
from pydtnn.utils.tensor import ChannelFormat, SampleFormat, TensorFormat, format_transpose

__all__ = ("Utils",)

logger = logging.getLogger(__name__)


type TransformFunc = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class Utils(Base):
    """
    Abstract base class for all datasets in PyDTNN.

    Defines the interface and common utilities for data partitioning,
    shape management, and format conversion.
    """

    @property
    def train_nsamples(self) -> int:
        """Get number of training samples."""
        return self._nsamples[Base.Part.TRAIN]

    @train_nsamples.setter
    def train_nsamples(self, value: int) -> None:
        """Set number of training samples."""
        self._nsamples[Base.Part.TRAIN] = value

    @property
    def val_nsamples(self) -> int:
        """Get number of validation samples."""
        return self._nsamples[Base.Part.VAL]

    @val_nsamples.setter
    def val_nsamples(self, value: int) -> None:
        """Set number of validation samples."""
        self._nsamples[Base.Part.VAL] = value

    @property
    def test_nsamples(self) -> int:
        """Get number of test samples."""
        return self._nsamples[Base.Part.TEST]

    @test_nsamples.setter
    def test_nsamples(self, value) -> None:
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
    def _offset2files(
        filenames: list[str], images_per_file: int, local_offset: int, local_nsamples: int
    ) -> list[tuple[str, int, int]]:
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

    def _gzip_open(self, filename: str) -> IO[bytes]:
        """
        Open a gZIP file.

        This method handles opening gZIP files, creating or loading their
        seek tables for efficient random access.

        Args:
            filename: The path to the gZIP file.

        Returns:
            A file-like object opened in binary read mode.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: If there's an error during rapidgzip file operations.
        """
        path = Path(filename)
        plain = path.with_suffix("")
        idx = path.with_suffix(f"{path.suffix}.idx")
        f = None

        if plain.exists():
            return open(plain, mode="rb")
        elif not path.exists():
            raise FileNotFoundError(path)
        try:
            f = rapidgzip.RapidgzipFile(path, parallelization=1)
            if idx.exists():
                f.import_index(str(idx))
            else:
                f.export_index(str(idx))
        except Exception:
            if f:
                f.close()
            raise
        else:
            return f

    def _load_rgb_image(self, fp: IO[bytes] | str) -> np.ndarray:
        """
        Transform a file-like object (RGB image) to a numpy array.

        Opens an image file, converts it to RGB format, and returns it as a
        numpy array with shape (C, H, W) and dtype uint8.

        Args:
            fp: A file-like object or a string path to the image file.

        Returns:
            A numpy array representing the RGB image in CHW format.
        """
        with Image.open(fp=fp) as image:
            image = image.convert("RGB")
            array = np.asarray(image, order="C")
            # NOTE: PIL mode RGB is WHC in unit8
            array = format_transpose(array, SampleFormat.WHC, SampleFormat.CHW)
        return array

    def _load_gray_image(self, fp: IO[bytes] | str) -> np.ndarray:
        """
        Transform a file-like object (gray-scale image) to a numpy array.

        Opens an image file, converts it to grayscale ('L' mode), and returns
        it as a numpy array with shape (1, H, W) and dtype uint8.

        Args:
            fp: A file-like object or a string path to the image file.

        Returns:
            A numpy array representing the grayscale image in CHW format.
        """
        with Image.open(fp=fp) as image:
            image = image.convert("L")
            array = np.asarray(image, order="C")
            # NOTE: PIL mode L is WH in unit8
            array = format_transpose(array, ChannelFormat.WH, ChannelFormat.HW)
            array = array[None, ...]
        return array
