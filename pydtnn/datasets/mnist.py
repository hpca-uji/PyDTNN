#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import os

import numpy as np

from .dataset import Dataset, TRAIN, TEST, VAL
from ..utils import PYDTNN_TENSOR_FORMAT_NHWC

TRAIN_NSAMPLES = 60000
TEST_NSAMPLES = 10000
INPUT_SHAPE = (1, 28, 28)
OUTPUT_SHAPE = (10,)


class MNIST(Dataset):

    def __init__(self, model):
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, INPUT_SHAPE, OUTPUT_SHAPE)

    def _init_actual_data(self):
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
        x_filename[VAL] = x_filename[TEST] if self.test_as_validation else x_filename[TRAIN]
        y_filename[VAL] = y_filename[TEST] if self.test_as_validation else y_filename[TRAIN]
        images_header_offset = 16  # 4 + 4 * 3
        labels_header_offset = 8  # 4 + 4 * 1
        for part in (TRAIN, VAL, TEST):
            offset = images_header_offset + self._local_offset[part] * np.prod(self.input_shape)
            nbytes = self._local_nsamples[part] * np.prod(self.input_shape)
            self._x[part] = self._read_file(x_filename[part], offset, nbytes) \
                                .reshape(self._local_nsamples[part], *self.input_shape) \
                                .astype(self.model.dtype) / 255.0
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NHWC:
                self._x[part] = self._nchw2nhwc(self._x[part])
            offset = labels_header_offset + self._local_offset[part] * 1  # The output class is encoded as a number
            nbytes = self._local_nsamples[part] * 1  # The output class is encoded as a number
            y_classes = self._read_file(y_filename[part], offset, nbytes)
            self._y[part] = np.zeros([self._local_nsamples[part]] + self.output_shape,
                                     dtype=self.model.dtype, order="C")
            self._decode_class(self._y[part], y_classes)

    @staticmethod
    def _read_file(filename, offset, nbytes):
        with open(filename, 'rb') as f:
            # How to read the header:
            #  zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            #  shape = (struct.unpack('>I', f.read(4))[0] for _ in range(dims))
            f.seek(offset)
            return np.frombuffer(f.read(nbytes), dtype=np.uint8)
