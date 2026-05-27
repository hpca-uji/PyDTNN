"""
PyDTNN Tsunami Dataset Module.

This module provides the Tsunamis dataset class, which is designed to load
and process tsunami simulation data for machine learning tasks.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import tarfile
from typing import TYPE_CHECKING, Generator

import numpy as np

from pydtnn.datasets.abstract import Dataset
from pydtnn.utils import random

__all__ = ("Tsunamis",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model

TRAIN_NSAMPLES = 50000
TEST_NSAMPLES = 10000
INPUT_SHAPE = (1, 80, 80)
OUTPUT_SHAPE = (1,)
IMAGES_PER_FILE = 10000


class Tsunamis(Dataset):
    """
    Tsunamis Dataset

    Source (SHA1): ???

    Normalize (z-score):
    offset: ???
    scale:  ???
    """

    def __init__(self, model: Model, force_test_as_validation=False, debug=False):
        """
        Initialize the Tsunamis dataset handler.

        Args:
            model: The model instance associated with the dataset.
            force_test_as_validation: Whether to use the test set as validation.
            debug: Whether to enable debug mode.
        """
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, INPUT_SHAPE, OUTPUT_SHAPE, force_test_as_validation=force_test_as_validation, debug=debug)

    def _model_init(self) -> None:
        """
        Initialize file paths and metadata for the tsunami dataset.
        """
        self._src_filename = os.path.join(self.model.dataset_path, "tsunamis-binary.tar.gz")
        self._xy_filenames = [[os.path.join("tsunamis-batches-bin", f"data_batch_{x}.bin") for x in range(1, 6)], [], [os.path.join("tsunamis-batches-bin", "test_batch.bin")]]
        self._xy_filenames[Dataset.Part.VAL] = copy.copy(self._xy_filenames[Dataset.Part.TEST] if self.test_as_validation else self._xy_filenames[Dataset.Part.TRAIN])

        # Pregenerate GZIP indexs
        self._gzip_open(self._src_filename).close()

    def _data_generator(self, part: Dataset.Part) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate batches of data for the specified dataset partition.

        Args:
            part: The dataset partition (TRAIN, VAL, or TEST).

        Yields:
            A tuple containing the input tensor and the target tensor.
        """
        xy_filenames = self._xy_filenames[part]

        if part is Dataset.Part.TRAIN and self.model.augment_shuffle:
            random.shuffle(xy_filenames)

        with self._gzip_open(self._src_filename) as g:
            with tarfile.open(fileobj=g) as t:
                for filename, offset, nsamples in self._offset2files(xy_filenames, IMAGES_PER_FILE, self._local_offset[part], self._local_nsamples[part]):
                    with t.extractfile(filename) as f:  # type: ignore
                        x, y_classes = self._read_file(f, offset, nsamples)

                    y = np.zeros((*y_classes.shape, *self.output_shape), dtype=self.model.dtype)
                    self._decode_class(y, y_classes)

                    x = self.model.encode_tensor(x)
                    x = np.divide(x, 255.0, dtype=self.model.dtype, casting="unsafe")

                    yield x, y

    def _read_file(self, f, offset, nsamples) -> tuple[np.ndarray, np.ndarray]:
        """
        Read a chunk of binary data from the file object.

        Args:
            f: The file object to read from.
            offset: The starting offset in the file.
            nsamples: The number of samples to read.

        Returns:
            A tuple containing the input images and their corresponding class labels.
        """
        chunk_size = math.prod(INPUT_SHAPE) + 1
        f.seek(offset * chunk_size)
        im = np.frombuffer(f.read(nsamples * chunk_size), dtype=np.uint8).reshape(nsamples, chunk_size)
        y_classes, x = im[:, 0].flatten(), im[:, 1:].reshape(nsamples, *INPUT_SHAPE).astype(self.model.dtype)
        return x, y_classes
