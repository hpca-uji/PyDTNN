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

import numpy as np
from .dataset import Dataset, TRAIN, VAL, TEST


class CustomDataset(Dataset):

    def __init__(self, model, x_train, y_train, x_test=None, y_test=None, input_shape=None, output_shape=None, force_test_as_validation=True):
        if x_test is None or y_test is None:
            if x_test is None and y_test is None:
                x_test = x_train
                y_test = y_train
            else:
                raise SystemExit("Both x_test and y_test must be provided or, alternatively, none of them!")

        if input_shape is None:
            input_shape = x_train.shape[1:]

        if output_shape is None:
            output_shape = y_train.shape[1:]

        self.__x_source = []
        self.__y_source = []
        # Sources for the training part
        self.__x_source.append(x_train)
        self.__y_source.append(y_train)
        # Sources for the validation part
        if force_test_as_validation:
            self.__x_source.append(x_test)
            self.__y_source.append(y_test)
        else:
            self.__x_source.append(x_train)
            self.__y_source.append(y_train)
        # Sources for the test part
        self.__x_source.append(x_test)
        self.__y_source.append(y_test)

        super().__init__(model,
                         x_train.shape[0],
                         x_test.shape[0],
                         input_shape,
                         output_shape,
                         force_test_as_validation=force_test_as_validation)

    def _init_actual_data(self):
        for part in (TRAIN, VAL, TEST):
            local_offset = self._local_offset[part]
            local_nsamples = self._local_nsamples[part]
            local_slice = slice(local_offset, local_offset + local_nsamples)
            self._x[part] = self.__x_source[part][local_slice, ...]
            self._y[part] = self.__y_source[part][local_slice, ...]

    @classmethod
    def import_(cls, model):
        """Import dataset (rank specific)"""
        with np.load(model.dataset_raw_path) as data:
            x_train = data["x_train"]
            y_train = data["y_train"]
            x_test = data["x_test"]
            y_test = data["y_test"]

            self = cls(
                model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                force_test_as_validation=False
            )

            # Debug information
            if self.debug:
                print(f"Import: {self.model.dataset_raw_path}")
                print(f"x_train: {x_train.shape}")
                print(f"y_train: {y_train.shape}")
                print(f"x_test: {x_test.shape}")
                print(f"y_test: {y_test.shape}")

            return self
