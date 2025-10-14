# Dataset source (SHA1):
# e8aa088b9774a44ad217101d2e2569f823d2d491 https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

import os
import math
import tarfile

import numpy as np

from pydtnn.utils.tensor import PYDTNN_TENSOR_FORMAT
from pydtnn.datasets.dataset import Dataset, DatasetEnum

from typing import TYPE_CHECKING, Generator
if TYPE_CHECKING:
    from pydtnn.model import Model

TRAIN_NSAMPLES = 50000
TEST_NSAMPLES = 10000
INPUT_SHAPE = (3, 32, 32)
OUTPUT_SHAPE = (10,)
IMAGES_PER_FILE = 10000


class CIFAR10(Dataset):
    """CIFAR10 Dataset"""

    def __init__(self, model: Model):
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, INPUT_SHAPE, OUTPUT_SHAPE, max_prefetch=math.ceil(model.batch_size / IMAGES_PER_FILE))

    def _init_actual_data(self):
        self._src_filename = self.model.dataset_path
        self._xy_filenames: list[str] = [
            [os.path.join("cifar-10-batches-bin", f"data_batch_{x}.bin") for x in range(1, 6)],
            [],
            [os.path.join("cifar-10-batches-bin", "test_batch.bin")]
        ]
        self._xy_filenames[DatasetEnum.VAL] = self._xy_filenames[DatasetEnum.TEST] if self.test_as_validation else self._xy_filenames[DatasetEnum.TRAIN]

    def _actual_data_generator(self, part: DatasetEnum):
        y_classes = np.array([])
        with tarfile.open(self._src_filename, "r:gz") as t:
            for part in (DatasetEnum.TRAIN, DatasetEnum.VAL, DatasetEnum.TEST):
                for filename, offset, nsamples in self._offset2files(self._xy_filenames[part], IMAGES_PER_FILE, self._local_offset[part], self._local_nsamples[part]):
                    with t.extractfile(filename) as f:
                        x, y_classes = self._read_file(f, offset, nsamples)
                    x /= 255.0

                    y = np.zeros((*y_classes.shape, *self.output_shape), dtype=self.model.dtype, order="C")
                    self._decode_class(y, y_classes)

                    if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NHWC:
                        x = self._nchw2nhwc(x)

                    yield x, y

    def _read_file(self, f, offset, nsamples):
        chunk_size = np.prod(self.real_input_shape) + 1
        f.seek(offset * chunk_size)
        im = np.frombuffer(f.read(nsamples * chunk_size), dtype=np.uint8).reshape(nsamples, chunk_size)
        y_classes, x = im[:, 0].flatten(), im[:, 1:].reshape(nsamples, *self.real_input_shape).astype(self.model.dtype, order="C")
        return x, y_classes
