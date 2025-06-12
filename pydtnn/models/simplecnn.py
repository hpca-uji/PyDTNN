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


# 4 2 1.5 1.2 1 0.9 0.75 0.6
# 40000

# 58 118 238 400
# 512

# 60 118 237 396
# 32 4096


# 4 7 15 26
# 4096/32 4096

# 1 2 3 4
# 175 375 40000

def create_simplecnn(input_shape: Sequence[int], output_shape: Sequence[int]) -> Iterable[LayerAndActivationBase]:
    yield Input(shape=input_shape)
    yield Conv2D(nfilters=4, filter_shape=(3, 3), padding=1, stride=1, activation="relu")
    yield Conv2D(nfilters=8, filter_shape=(3, 3), padding=1, stride=1, activation="relu")
    yield MaxPool2D(pool_shape=(2, 2), stride=2)
    yield Flatten()
    for i in range(1):
        yield FC(shape=(int(175),), activation="relu")
        yield FC(shape=(int(375),), activation="relu")
        yield FC(shape=(int(40000),), activation="relu")
    yield Dropout(rate=0.5)
    yield FC(shape=output_shape, activation="softmax")
