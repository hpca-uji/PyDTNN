# https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
# https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
# https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
# https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

import os

import numpy as np

from pydtnn.datasets.dataset import Dataset, DatasetEnum
from pydtnn.utils import PYDTNN_TENSOR_FORMAT

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model

TRAIN_NSAMPLES = 60000
TEST_NSAMPLES = 10000
INPUT_SHAPE = (1, 28, 28)
OUTPUT_SHAPE = (10,)


class MNIST(Dataset):

    def __init__(self, model: "Model"):
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, INPUT_SHAPE, OUTPUT_SHAPE)

    def _init_actual_data(self) -> None:
        x_filename = [
            os.path.join(self.model.dataset_train_path, "train-images-idx3-ubyte"),
            None,
            os.path.join(self.model.dataset_test_path, "t10k-images-idx3-ubyte")
        ]
        y_filename = [
            os.path.join(self.model.dataset_train_path, "train-labels-idx1-ubyte"),
            None,
            os.path.join(self.model.dataset_test_path, "t10k-labels-idx1-ubyte")
        ]
        x_filename[DatasetEnum.VAL] = x_filename[DatasetEnum.TEST] if self.test_as_validation else x_filename[DatasetEnum.TRAIN]
        y_filename[DatasetEnum.VAL] = y_filename[DatasetEnum.TEST] if self.test_as_validation else y_filename[DatasetEnum.TRAIN]
        images_header_offset = 16  # 4 + 4 * 3
        labels_header_offset = 8  # 4 + 4 * 1
        for part in (DatasetEnum.TRAIN, DatasetEnum.VAL, DatasetEnum.TEST):
            offset = images_header_offset + self._local_offset[part] * np.prod(self.real_input_shape)
            nbytes = self._local_nsamples[part] * np.prod(self.real_input_shape)
            self._x[part] = self._read_file(x_filename[part], offset, nbytes) \
                                .reshape(self._local_nsamples[part], *self.real_input_shape) / 255.0
            self._x[part] = self._x[part].astype(self.model.dtype)
            if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NHWC:
                self._x[part] = self._nchw2nhwc(self._x[part])
            offset = labels_header_offset + self._local_offset[part] * 1  # The output class is encoded as a number
            nbytes = self._local_nsamples[part] * 1  # The output class is encoded as a number
            y_classes = self._read_file(y_filename[part], offset, nbytes)
            self._y[part] = np.zeros([self._local_nsamples[part]] + self.output_shape,
                                     dtype=self.model.dtype, order="C")
            self._decode_class(self._y[part], y_classes)

    @staticmethod
    def _read_file(filename, offset: int, nbytes: int) -> np.ndarray:
        with open(filename, 'rb') as f:
            # How to read the header:
            #  zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            #  shape = (struct.unpack('>I', f.read(4))[0] for _ in range(dims))
            f.seek(offset)
            return np.frombuffer(f.read(nbytes), dtype=np.uint8)
