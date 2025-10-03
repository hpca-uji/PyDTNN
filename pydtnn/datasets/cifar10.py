# https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

import os

import numpy as np

from pydtnn.utils import PYDTNN_TENSOR_FORMAT
from pydtnn.datasets.dataset import Dataset, DatasetEnum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model
else:
    Model = object

TRAIN_NSAMPLES = 50000
TEST_NSAMPLES = 10000
INPUT_SHAPE = (3, 32, 32)
OUTPUT_SHAPE = (10,)
IMAGES_PER_FILE = 10000


class CIFAR10(Dataset):
    """CIFAR10 Dataset"""

    def __init__(self, model: Model):
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, INPUT_SHAPE, OUTPUT_SHAPE)

    def _init_actual_data(self):
        xy_filenames: list[str] = [
            [os.path.join(self.model.dataset_train_path, f"data_batch_{x}.bin") for x in range(1, 6)],
            [],
            [os.path.join(self.model.dataset_test_path, "test_batch.bin")]
        ]
        xy_filenames[DatasetEnum.VAL] = xy_filenames[DatasetEnum.TEST] if self.test_as_validation else xy_filenames[DatasetEnum.TRAIN]
        y_classes = np.array([])
        for part in (DatasetEnum.TRAIN, DatasetEnum.VAL, DatasetEnum.TEST):
            for filename, offset, nsamples in self._offset2files(xy_filenames[part],
                                                                 IMAGES_PER_FILE,
                                                                 self._local_offset[part],
                                                                 self._local_nsamples[part]):
                x, y_classes = self._read_file(filename, offset, nsamples)
                self._x[part] = np.concatenate((self._x[part], x), axis=0)
                y = np.zeros(list(y_classes.shape) + self.output_shape,
                             dtype=self.model.dtype, order="C")
                self._decode_class(y, y_classes)
                self._y[part] = np.concatenate((self._y[part], y), axis=0)
            self._x[part] = self._x[part] / 255.0
            self._x[part] = self._normalize_image(self._x[part])
            if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NHWC:
                self._x[part] = self._nchw2nhwc(self._x[part])

    def _read_file(self, filename, offset, nsamples):
        with open(filename, 'rb') as f:
            chunk_size = np.prod(self.input_shape) + 1
            f.seek(offset * chunk_size)
            im = np.frombuffer(f.read(nsamples * chunk_size), dtype=np.uint8).reshape(nsamples, chunk_size)
            y_classes, x = im[:, 0].flatten(), im[:, 1:].reshape(nsamples, *self.input_shape).astype(self.model.dtype)
            return x, y_classes

    # @staticmethod
    def _normalize_image(self, x):
        if not hasattr(self, "mean"):
            self.mean = np.mean(x, axis=(0, 2, 3))
            self.std = np.std(x, axis=(0, 2, 3))
        for c in range(3):
            x[:, c, ...] = (x[:, c, ...] - self.mean[c]) / self.std[c]
        return x
