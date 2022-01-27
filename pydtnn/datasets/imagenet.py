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

from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NHWC
from .dataset import Dataset, TRAIN, TEST, VAL

# The most highly-used subset of ImageNet is the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012-2017
# image classification and localization dataset. This dataset spans 1000 object classes and contains 1,281,
# 167 training images, 50,000 validation images and 100,000 test images. This subset is available on Kaggle.

TRAIN_NSAMPLES = 1281167
TEST_NSAMPLES = 100000
INPUT_SHAPE = (3, 227, 227)
OUTPUT_SHAPE = (1000,)
IMAGES_PER_TRAIN_FILE = 1251
IMAGES_PER_TEST_FILE = 390


class ImageNet(Dataset):

    def __init__(self, model):
        # for VGG, ResNet and other models input shape must be (3,224,224)
        input_shape = INPUT_SHAPE if "alexnet" in model.model_name else (3, 224, 224)
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, input_shape, OUTPUT_SHAPE)
        self._xy_filenames = [[] for _ in range(3)] # warning: [[]] * 3 replicates the same array
        self._images_per_file = [[] for _ in range(3)]  # warning: [[]] * 3 replicates the same array
        if not self.model.use_synthetic_data:
            # Train part
            path = self.model.dataset_train_path
            self._xy_filenames[TRAIN] = [os.path.join(path, f) for f in sorted(os.listdir(path))]
            self._images_per_file[TRAIN] = IMAGES_PER_TRAIN_FILE
            # Test part
            path = self.model.dataset_test_path
            self._xy_filenames[TEST] = [os.path.join(path, f) for f in sorted(os.listdir(path))]
            self._images_per_file[TEST] = IMAGES_PER_TEST_FILE
            # Validation part
            self._xy_filenames[VAL] = self._xy_filenames[TEST] if self.test_as_validation else self._xy_filenames[TRAIN]
            self._images_per_file[VAL] = self._images_per_file[TEST] \
                if self.test_as_validation else self._images_per_file[TRAIN]

    def _init_actual_data(self):
        # There is no initialization, as the data is huge, it will be read from the corresponding files as required
        pass

    def _normalize_image(self, x):
        if "alexnet" not in self.model.model_name:  # for VGG, ResNet and other models input shape must be (3,224,224)
            x = x[..., 1:225, 1:225]
        # A) Caffe-like normalization used for pre-trained Keras models
        x = x[:, ::-1, ...]
        mean = [103.939, 116.779, 123.68]
        for c in range(3):
            x[:, c, :, :] -= mean[c]
        # std = [?, ?, ?]
        # for c in range(3):
        #     x[:, c, :, :] /= std[c]

        # B) Alternative normalization
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # for c in range(3):
        #     x[:, c, ...] = ((x[:, c, ...] / 255.0) - mean[c]) / std[c]
        return x

    def _actual_data_generator(self, part):
        files = self._xy_filenames[part]
        images_per_file = self._images_per_file[part]
        if part == TRAIN:
            # Note that the actual array is shuffled. This is done to ensure that the validation part,
            # when is extracted from the train samples, uses the same files order.
            np.random.shuffle(files)
        max_nsamples_online = self.max_batches_online * self.model.batch_size
        for filename, offset, nsamples in self._offset2files(files,
                                                             images_per_file,
                                                             self._local_offset[part],
                                                             self._local_nsamples[part]):
            # Extract x[offset:offset+nsamples] and y[offset:offset+nsamples] from file
            values = np.load(filename)
            x = self._normalize_image(values['x'][offset:offset + nsamples].astype(self.model.dtype))
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NHWC:
                x = self._nchw2nhwc(x)
            y = np.zeros(list(x.shape[0]) + self.output_shape, dtype=self.model.dtype, order="C")
            self._decode_class(y, values['y'][offset:offset + nsamples].astype(np.int16).flatten() - 1)

            # Concatenate x and y with current _x[part] and _y[part]
            self._x[part] = np.concatenate((self._x[part], x), axis=0)
            self._y[part] = np.concatenate((self._y[part], y), axis=0)

            # Consume as many max_batches_online as possible
            while self._x[part].shape[0] >= max_nsamples_online:
                x_data = self._x[part][:max_nsamples_online, ...]
                y_data = self._y[part][:max_nsamples_online, ...]
                self._x[part] = self._x[part][max_nsamples_online:, ...]
                self._y[part] = self._y[part][max_nsamples_online:, ...]
                yield x_data, y_data
        # After reading all the files, consume the remaining data and empty _x[part] and _y[part]
        if self._x[part].shape[0] > 0:
            x_data = self._x[part][...]
            y_data = self._y[part][...]
            self._x[part] = self._x[part][:0, ...]
            self._y[part] = self._y[part][:0, ...]
            yield x_data, y_data
