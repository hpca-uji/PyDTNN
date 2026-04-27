import copy
import logging
import math
import os
import tarfile
from typing import TYPE_CHECKING

import numpy as np

from pydtnn.datasets.dataset import Dataset
from pydtnn.utils import random

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model

TRAIN_NSAMPLES = 50000
TEST_NSAMPLES = 10000
INPUT_SHAPE = (3, 32, 32)
OUTPUT_SHAPE = (10,)
IMAGES_PER_FILE = 10000


class CIFAR10(Dataset):
    """
    CIFAR10 Dataset

    Database of the 80 million tiny images dataset.

    Source (SHA1): https://www.cs.toronto.edu/~kriz/cifar.html
    e8aa088b9774a44ad217101d2e2569f823d2d491 cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

    Normalize (z-score):
    offset: -0.475
    scale:  +3.964
    """

    def __init__(self, model: "Model", force_test_as_validation=False, debug=False):
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, INPUT_SHAPE, OUTPUT_SHAPE, force_test_as_validation=force_test_as_validation, debug=debug)

    def _init_actual_data(self):
        self._src_filename = os.path.join(self.model.dataset_path, "cifar-10-binary.tar.gz")
        self._xy_filenames = [
            [os.path.join("cifar-10-batches-bin", f"data_batch_{x}.bin") for x in range(1, 6)],
            [],
            [os.path.join("cifar-10-batches-bin", "test_batch.bin")]
        ]
        self._xy_filenames[Dataset.Part.VAL] = copy.copy(self._xy_filenames[Dataset.Part.TEST] if self.test_as_validation else self._xy_filenames[Dataset.Part.TRAIN])

        # Pregenerate GZIP indexs
        self._gzip_open(self._src_filename).close()

    def _actual_data_generator(self, part: Dataset.Part):
        xy_filenames = self._xy_filenames[part]

        if part is Dataset.Part.TRAIN and self.model.augment_shuffle:
            random.shuffle(xy_filenames)

        with self._gzip_open(self._src_filename) as g:
            with tarfile.open(fileobj=g) as t:
                for filename, offset, nsamples in self._offset2files(xy_filenames, IMAGES_PER_FILE, self._local_offset[part], self._local_nsamples[part]):
                    with t.extractfile(filename) as f:  # type: ignore
                        x, y_classes = self._read_file(f, offset, nsamples)
                    x = np.divide(x, 255.0, dtype=self.model.dtype, casting="unsafe")

                    y = np.zeros((*y_classes.shape, *self.output_shape), dtype=self.model.dtype)
                    self._decode_class(y, y_classes)

                    x = self.model.encode_tensor(x)

                    yield x, y

    def _read_file(self, f, offset, nsamples):
        chunk_size = math.prod(INPUT_SHAPE) + 1
        f.seek(offset * chunk_size)
        im = np.frombuffer(f.read(nsamples * chunk_size), dtype=np.uint8).reshape(nsamples, chunk_size)
        y_classes, x = im[:, 0].flatten(), im[:, 1:].reshape(nsamples, *INPUT_SHAPE).astype(self.model.dtype)
        return x, y_classes
