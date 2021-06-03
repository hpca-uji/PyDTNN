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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from ..layers import *


def create_vgg1(model):
    _ = model.add
    _(Input(shape=(32, 32, 3)))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, activation="relu", weights_initializer="he_uniform"))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, activation="relu", weights_initializer="he_uniform"))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(128,), activation="relu", weights_initializer="he_uniform"))
    _(FC(shape=(10,), activation="softmax"))
