import os
import gzip

import numpy as np

from pydtnn.datasets.dataset import Dataset
from pydtnn.utils.tensor import PYDTNN_TENSOR_FORMAT

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model

TRAIN_NSAMPLES = 60000
TEST_NSAMPLES = 10000
INPUT_SHAPE = (1, 28, 28)
OUTPUT_SHAPE = (10,)


class MNIST(Dataset):
    """
    MNIST Dataset

    Source (SHA1):
    6c95f4b05d2bf285e1bfb0e7960c31bd3b3f8a7d https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    2a80914081dc54586dbdf242f9805a6b8d2a15fc https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    c3a25af1f52dad7f726cce8cacb138654b760d48 https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    763e7fa3757d93b0cdec073cef058b2004252c17 https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

    Normalize:
    offset: -0.131
    scale:   3.245
    """
    # mean: [0.1307281]
    # std:  [0.30818242]

    def __init__(self, model: "Model", force_test_as_validation=False, debug=False):
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, INPUT_SHAPE, OUTPUT_SHAPE, force_test_as_validation=force_test_as_validation, debug=debug)

    def _init_actual_data(self) -> None:
        self._x_filename = [
            os.path.join(self.model.dataset_train_path, "train-images-idx3-ubyte.gz"),
            None,
            os.path.join(self.model.dataset_test_path, "t10k-images-idx3-ubyte.gz")
        ]
        self._y_filename = [
            os.path.join(self.model.dataset_train_path, "train-labels-idx1-ubyte.gz"),
            None,
            os.path.join(self.model.dataset_test_path, "t10k-labels-idx1-ubyte.gz")
        ]
        if self.test_as_validation:
            self._x_filename[Dataset.Part.VAL] = self._x_filename[Dataset.Part.TEST]
            self._y_filename[Dataset.Part.VAL] = self._y_filename[Dataset.Part.TEST]
        else:
            self._x_filename[Dataset.Part.VAL] = self._x_filename[Dataset.Part.TRAIN]
            self._y_filename[Dataset.Part.VAL] = self._y_filename[Dataset.Part.TRAIN]

        self._images_header_offset = 16  # 4 + 4 * 3
        self._labels_header_offset = 8  # 4 + 4 * 1

    def _actual_data_generator(self, part: Dataset.Part):
        for part in Dataset.Part:
            offset = self._images_header_offset + self._local_offset[part] * np.prod(INPUT_SHAPE)
            nbytes = self._local_nsamples[part] * np.prod(INPUT_SHAPE)
            with gzip.open(self._x_filename[part], "rb") as f:
                x = self._read_file(f, offset, nbytes).reshape(self._local_nsamples[part], *INPUT_SHAPE) / 255.0
            x = x.astype(self.model.dtype)

            if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NHWC:
                x = self._nchw2nhwc(x)

            offset = self._labels_header_offset + self._local_offset[part] * 1  # The output class is encoded as a number
            nbytes = self._local_nsamples[part] * 1  # The output class is encoded as a number

            with gzip.open(self._y_filename[part], "rb") as f:
                y_classes = self._read_file(f, offset, nbytes)

            y = np.zeros((self._local_nsamples[part], *self.output_shape), dtype=self.model.dtype, order="C")
            self._decode_class(y, y_classes)

            yield x, y

    def _read_file(self, f, offset: int, nbytes: int) -> np.ndarray:
        # How to read the header:
        #  zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        #  shape = (struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        f.seek(offset)
        return np.frombuffer(f.read(nbytes), dtype=np.uint8)
