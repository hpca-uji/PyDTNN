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
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from ..activations import relu, softmax

def create_alexnet(input_shape: Sequence[int], output_shape: Sequence[int]) -> Iterable[LayerAndActivationBase]:
    yield Input(shape = input_shape)
    yield Conv2D(nfilters=96, filter_shape=(11, 11), padding=0, stride=4, activation=relu)
    yield MaxPool2D(pool_shape=(3, 3), stride=2)
    yield Conv2D(nfilters=256, filter_shape=(5, 5), padding=2, stride=1, activation=relu)
    yield MaxPool2D(pool_shape=(3, 3), stride=2)
    yield Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation=relu)
    yield Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation=relu)
    yield Conv2D(nfilters=256, filter_shape=(3, 3), padding=1, stride=1, activation=relu)
    yield MaxPool2D(pool_shape=(3, 3), stride=2)
    yield Flatten()
    yield FC(shape=(4096,), activation=relu)
    yield Dropout(rate=0.5)
    yield FC(shape=(4096,), activation=relu)
    yield Dropout(rate=0.5)
    yield FC(shape = output_shape, activation=softmax)
