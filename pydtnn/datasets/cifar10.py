"""CIFAR-10 dataset implementation for PyDTNN."""

from __future__ import annotations

import logging
import math
import os
import tarfile
from typing import IO, TYPE_CHECKING, Generator

import numpy as np

from pydtnn.datasets.abstract import Dataset
from pydtnn.datasets.memory import Memory

__all__ = ("CIFAR10",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model

TRAIN_NSAMPLES = 50000
TEST_NSAMPLES = 10000
INPUT_SHAPE = (3, 32, 32)
OUTPUT_SHAPE = (10,)
IMAGES_PER_FILE = 10000


class CIFAR10(Memory):
    """
    CIFAR10 Dataset

    Database of the 80 million tiny images dataset.

    Source (SHA1): https://www.cs.toronto.edu/~kriz/cifar.html
    e8aa088b9774a44ad217101d2e2569f823d2d491 cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

    Normalize (z-score):
    offset: -0.475
    scale:  +3.964
    """

    def __init__(
        self, model: Model, force_test_as_validation: bool = False, debug: bool = False
    ) -> None:
        """
        Initialize the CIFAR10 dataset.

        Args:
            model: The model instance associated with the dataset.
            force_test_as_validation: Whether to use the test set as validation.
            debug: Whether to enable debug mode.
        """
        src_filename = os.path.join(model.dataset_path, "cifar-10-binary.tar.gz")
        train_filenames = [
            os.path.join("cifar-10-batches-bin", f"data_batch_{x}.bin") for x in range(1, 6)
        ]
        test_filenames = [os.path.join("cifar-10-batches-bin", "test_batch.bin")]

        x_parts = []
        y_parts = []
        with self._gzip_open(src_filename) as g, tarfile.open(fileobj=g) as t:
            for filename, offset, nsamples in self._offset2files(
                train_filenames,
                IMAGES_PER_FILE,
                0,
                TRAIN_NSAMPLES,
            ):
                with t.extractfile(filename) as f:  # pyright: ignore[reportOptionalContextManager]
                    x_part, y_class = self._read_file(f, offset, nsamples)
                    x_parts.append(x_part)

                    y_part = np.zeros((*y_class.shape, *OUTPUT_SHAPE), dtype=np.uint8)
                    self._decode_class(y_part, y_class)
                    y_parts.append(y_part)
        x_train = np.concatenate(x_parts)
        y_train = np.concatenate(y_parts)
        x_parts.clear()
        y_parts.clear()

        x_parts = []
        y_parts = []
        with self._gzip_open(src_filename) as g, tarfile.open(fileobj=g) as t:
            for filename, offset, nsamples in self._offset2files(
                test_filenames,
                IMAGES_PER_FILE,
                0,
                TEST_NSAMPLES,
            ):
                with t.extractfile(filename) as f:  # pyright: ignore[reportOptionalContextManager]
                    x_part, y_class = self._read_file(f, offset, nsamples)
                    x_parts.append(x_part)

                    y_part = np.zeros((*y_class.shape, *OUTPUT_SHAPE), dtype=np.uint8)
                    self._decode_class(y_part, y_class)
                    y_parts.append(y_part)
        x_test = np.concatenate(x_parts)
        y_test = np.concatenate(y_parts)
        x_parts.clear()
        y_parts.clear()

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
        Generate batches of CIFAR10 data.

        Args:
            part: The dataset partition to generate.
        """
        for x, y in super()._data_generator(part):
            x = self.model.encode_tensor(x)
            x = np.divide(x, 255.0, dtype=self.model.dtype, casting="unsafe")
            y = np.asarray(y, dtype=self.model.dtype)
            yield x, y

    def _read_file(self, f: IO, offset: int, nsamples: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Read raw binary data from the CIFAR-10 file format.

        Args:
            f: File-like object.
            offset: Number of samples to skip.
            nsamples: Number of samples to read.

        Returns:
            A tuple of (images, labels).
        """
        chunk_size = math.prod(INPUT_SHAPE) + 1
        f.seek(offset * chunk_size)
        im = np.frombuffer(f.read(nsamples * chunk_size), dtype=np.uint8).reshape(
            nsamples, chunk_size
        )
        y_classes, x = (
            im[:, 0].flatten(),
            im[:, 1:].reshape(nsamples, *INPUT_SHAPE),
        )
        return x, y_classes
