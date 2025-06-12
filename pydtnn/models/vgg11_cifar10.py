#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
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

from collections.abc import Sequence, Iterable

from ..layers import *
from layers.layer import LayerAndActivationBase


def create_vgg11_cifar10(input_shape: Sequence[int], output_shape: Sequence[int]) -> Iterable[LayerAndActivationBase]:
    yield Input(shape=input_shape)
    conv_pattern = [[1, 64], [1, 128], [2, 256], [2, 512], [2, 512]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            yield Conv2D(nfilters=nfilters, filter_shape=(3, 3, padding=1, stride=1, activation="relu",
                     weights_initializer="he_uniform"))
        yield MaxPool2D(pool_shape=(2, 2), stride=2)
    yield Flatten()
    yield FC(shape=(512,), activation="relu", weights_initializer="he_uniform")
    yield Dropout(rate=0.5)
    yield FC(shape=(512,), activation="relu", weights_initializer="he_uniform")
    yield Dropout(rate=0.5)
    yield FC(shape=output_shape, activation="softmax", weights_initializer="he_uniform")
