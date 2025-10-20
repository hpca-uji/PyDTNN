# Dataset source (SHA1):
# e8aa088b9774a44ad217101d2e2569f823d2d491 https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

import os
import copy
import math
import tarfile

import numpy as np

from pydtnn.utils.tensor import PYDTNN_TENSOR_FORMAT
from pydtnn.datasets.dataset import Dataset
from pydtnn.utils import random

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model

TRAIN_NSAMPLES = 50000
TEST_NSAMPLES = 10000
INPUT_SHAPE = (3, 32, 32)
OUTPUT_SHAPE = (10,)
IMAGES_PER_FILE = 10000


class CIFAR10(Dataset):
    """CIFAR10 Dataset"""
    # mean: [0.48995113 0.4807823  0.4451906 ]
    # std:  [0.24761744 0.2437481  0.26142704]

    def __init__(self, model: "Model", force_test_as_validation=False, debug=False):
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, INPUT_SHAPE, OUTPUT_SHAPE, max_prefetch=math.ceil(model.batch_size / IMAGES_PER_FILE), force_test_as_validation=force_test_as_validation, debug=debug)

    def _init_actual_data(self):
        self._src_filename = self.model.dataset_path
        self._xy_filenames = [
            [os.path.join("cifar-10-batches-bin", f"data_batch_{x}.bin") for x in range(1, 6)],
            [],
            [os.path.join("cifar-10-batches-bin", "test_batch.bin")]
        ]
        self._xy_filenames[Dataset.Part.VAL] = copy.copy(self._xy_filenames[Dataset.Part.TEST] if self.test_as_validation else self._xy_filenames[Dataset.Part.TRAIN])

    def _actual_data_generator(self, part: Dataset.Part):
        xy_filenames = self._xy_filenames[part]

        if part is Dataset.Part.TRAIN:
            random.shuffle(xy_filenames)

        with tarfile.open(self._src_filename, "r:gz") as t:
            for filename, offset, nsamples in self._offset2files(xy_filenames, IMAGES_PER_FILE, self._local_offset[part], self._local_nsamples[part]):
                with t.extractfile(filename) as f:
                    x, y_classes = self._read_file(f, offset, nsamples)
                x /= 255.0
                # x = self._normalize_image(x)

                y = np.zeros((*y_classes.shape, *self.output_shape), dtype=self.model.dtype, order="C")
                self._decode_class(y, y_classes)

                if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NHWC:
                    x = self._nchw2nhwc(x)

                yield x, y

    def _read_file(self, f, offset, nsamples):
        chunk_size = np.prod(INPUT_SHAPE) + 1
        f.seek(offset * chunk_size)
        im = np.frombuffer(f.read(nsamples * chunk_size), dtype=np.uint8).reshape(nsamples, chunk_size)
        y_classes, x = im[:, 0].flatten(), im[:, 1:].reshape(nsamples, *INPUT_SHAPE).astype(self.model.dtype, order="C")
        return x, y_classes

    def _normalize_image(self, x):
        print(f"{x.shape=}")
        mean = np.mean(x, axis=(0, 2, 3))
        std = np.std(x, axis=(0, 2, 3))
        print(f"{mean.shape=}")
        for c in range(3):
            x[:, c, ...] = (x[:, c, ...] - mean[c]) / std[c]
        return x
