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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from ..layers import *


def create_alexnet(model):
    _ = model.add
    _(Input(shape=(227, 227, 3)))
    _(Conv2D(nfilters=96, filter_shape=(11, 11), padding=0, stride=4, activation="relu"))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))
    _(Conv2D(nfilters=256, filter_shape=(5, 5), padding=2, stride=1, activation="relu"))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))
    _(Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation="relu"))
    _(Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation="relu"))
    _(Conv2D(nfilters=256, filter_shape=(3, 3), padding=1, stride=1, activation="relu"))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))
    _(Flatten())
    _(FC(shape=(4096,), activation="relu"))
    _(Dropout(rate=0.5))
    _(FC(shape=(4096,), activation="relu"))
    _(Dropout(rate=0.5))
    _(FC(shape=(1000,), activation="softmax"))
