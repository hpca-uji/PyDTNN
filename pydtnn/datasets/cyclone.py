"""
Cyclone dataset implementation for the PyDTNN framework.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import tarfile
from typing import TYPE_CHECKING

import numpy as np

from pydtnn.datasets.abstract import Dataset
from pydtnn.utils import random

__all__ = ("Cyclone",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model

TRAIN_NSAMPLES = 50000
TEST_NSAMPLES = 10000
INPUT_SHAPE = (5, 40, 40)
OUTPUT_SHAPE = (2,)
IMAGES_PER_FILE = 10000


class Cyclone(Dataset):
    """
    Cyclone Dataset

    Source (SHA1): ???

    Normalize (z-score):
    offset: ???
    scale:  ???
    """

    def __init__(self, model: Model, force_test_as_validation=False, debug=False):
        """
        Initialize the Cyclone dataset.

        Args:
            model: The model instance associated with the dataset.
            force_test_as_validation: Whether to use test data for validation.
            debug: Whether to enable debug mode.
        """
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, INPUT_SHAPE, OUTPUT_SHAPE, force_test_as_validation=force_test_as_validation, debug=debug)

    def _model_init(self):
        """
        Initialize file paths and verify dataset archive accessibility.
        """
        self._src_filename = os.path.join(self.model.dataset_path, "cyclone-binary.tar.gz")
        self._xy_filenames = [[os.path.join("cyclone-batches-bin", f"data_batch_{x}.bin") for x in range(1, 6)], [], [os.path.join("cyclone-batches-bin", "test_batch.bin")]]
        self._xy_filenames[Dataset.Part.VAL] = copy.copy(self._xy_filenames[Dataset.Part.TEST] if self.test_as_validation else self._xy_filenames[Dataset.Part.TRAIN])

        # Pregenerate GZIP indexs
        self._gzip_open(self._src_filename).close()

    def _data_generator(self, part: Dataset.Part):
        """
        Generate batches of data for the specified dataset partition.

        Args:
            part: The dataset partition (TRAIN, VAL, or TEST).

        Yields:
            A tuple containing the input tensor and the target labels.
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

    def _read_file(self, f, offset, nsamples):
        """
        Read a chunk of data from the binary file.

        Args:
            f: The file object to read from.
            offset: The starting index within the file.
            nsamples: The number of samples to read.

        Returns:
            A tuple containing the input data array and the class labels array.
        """
        chunk_size = math.prod(INPUT_SHAPE) + 1
        f.seek(offset * chunk_size)
        im = np.frombuffer(f.read(nsamples * chunk_size), dtype=np.uint8).reshape(nsamples, chunk_size)
        y_classes, x = im[:, 0].flatten(), im[:, 1:].reshape(nsamples, *INPUT_SHAPE)
        return x, y_classes
