#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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

from .dataset import Dataset, TRAIN, VAL, TEST


class CustomDataset(Dataset):

    def __init__(self, model, x_test, y_test, x_train=None, y_train=None):
        if x_train is None or y_train is None:
            if x_train is None and y_train is None:
                x_train = x_test
                y_train = y_test
            else:
                raise SystemExit("Both x_train and y_train must be provided or, alternatively, none of them!")
        super().__init__(model, x_train.shape[0], x_test.shape[0], x_train.shape[1:], y_test.shape[1:],
                         force_test_as_validation=True)
        self.__x_source = []
        self.__y_source = []
        # Sources for the training part
        self.__x_source.append(x_train)
        self.__y_source.append(y_train)
        # Sources for the validation part
        self.__x_source.append(x_test)
        self.__y_source.append(y_test)
        # Sources for the test part
        self.__x_source.append(x_test)
        self.__y_source.append(y_test)

    def _init_actual_data(self):
        for part in (TRAIN, VAL, TEST):
            self._x[part] = self.__x_source[part][self._local_offset[part]:self._local_nsamples[part], ...]
            self._y[part] = self.__y_source[part][self._local_offset[part]:self._local_nsamples[part], ...]
