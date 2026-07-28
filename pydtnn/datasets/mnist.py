"""MNIST dataset implementation for PyDTNN."""

from __future__ import annotations

import logging
import math
import os
from typing import IO, TYPE_CHECKING
from collections.abc import Generator

import numpy as np

from pydtnn.datasets.abstract import Dataset
from pydtnn.datasets.memory import Memory

__all__ = ("MNIST",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model

TRAIN_NSAMPLES = 60000
TEST_NSAMPLES = 10000
INPUT_SHAPE = (1, 28, 28)
OUTPUT_SHAPE = (10,)


class MNIST(Memory):
    """
    MNIST Dataset

    Handwritten digit database.

    Source (SHA1):
    6c95f4b05d2bf285e1bfb0e7960c31bd3b3f8a7d train-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    2a80914081dc54586dbdf242f9805a6b8d2a15fc train-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    c3a25af1f52dad7f726cce8cacb138654b760d48 t10k-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    763e7fa3757d93b0cdec073cef058b2004252c17 t10k-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    """  # noqa: E501

    # z-score
    normal_offset: float = -0.131
    normal_scale: float = +3.237

    def __init__(
        self, model: Model, force_test_as_validation: bool = False, debug: bool = False
    ) -> None:
        """
        Initialize the MNIST dataset.

        Args:
            model: The model instance.
            force_test_as_validation: Whether to use test set as validation set.
            debug: Whether to enable debug mode.
        """
        images_header_offset = 16  # 4 + 4 * 3
        labels_header_offset = 8  # 4 + 4 * 1

        x_train_filename = os.path.join(model.dataset_path, "train-images-idx3-ubyte.gz")
        y_train_filename = os.path.join(model.dataset_path, "train-labels-idx1-ubyte.gz")
        x_test_filename = os.path.join(model.dataset_path, "t10k-images-idx3-ubyte.gz")
        y_test_filename = os.path.join(model.dataset_path, "t10k-labels-idx1-ubyte.gz")

        with self._gzip_open(x_train_filename) as f:
            size = math.prod(INPUT_SHAPE)
            offset = images_header_offset + 0 * size
            x_train = self._read_file(f, offset, size * TRAIN_NSAMPLES).reshape(
                (TRAIN_NSAMPLES, *INPUT_SHAPE)
            )

        y_train = np.zeros((TRAIN_NSAMPLES, *OUTPUT_SHAPE), dtype=np.uint8)
        with self._gzip_open(y_train_filename) as f:
            size = 1
            offset = labels_header_offset + 0 * size
            y_class = self._read_file(f, offset, size * TRAIN_NSAMPLES)
            self._decode_class(y_train, y_class)

        with self._gzip_open(x_test_filename) as f:
            size = math.prod(INPUT_SHAPE)
            offset = images_header_offset + 0 * size
            x_test = self._read_file(f, offset, size * TEST_NSAMPLES).reshape(
                (TEST_NSAMPLES, *INPUT_SHAPE)
            )

        y_test = np.zeros((TEST_NSAMPLES, *OUTPUT_SHAPE), dtype=np.uint8)
        with self._gzip_open(y_test_filename) as f:
            size = 1
            offset = labels_header_offset + 0 * size
            y_class = self._read_file(f, offset, size * TEST_NSAMPLES)
            self._decode_class(y_test, y_class)

        super().__init__(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            INPUT_SHAPE,
            OUTPUT_SHAPE,
            force_test_as_validation=force_test_as_validation,
            debug=debug,
        )

    def _data_generator(self, part: Dataset.Part) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate batches of MNIST data.

        Args:
            part: The dataset partition to generate.
        """
        for x, y in super()._data_generator(part):
            x = self.model.encode_tensor(x)
            x = np.divide(x, 255.0, dtype=self.model.dtype, casting="unsafe")
            y = np.asarray(y, dtype=self.model.dtype)
            yield x, y

    def _read_file(self, f: IO[bytes], offset: int, nbytes: int) -> np.ndarray:
        """
        Read raw bytes from a file at a specific offset.

        Args:
            f: The file object.
            offset: The byte offset to seek to.
            nbytes: The number of bytes to read.

        Returns:
            A numpy array containing the read data.
        """
        # How to read the header:
        #  zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        #  shape = (struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        f.seek(offset)
        return np.frombuffer(f.read(nbytes), dtype=np.uint8)
