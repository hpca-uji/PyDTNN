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

from collections.abc import Sequence

from ..layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase


def create_simplecnn(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=4, filter_shape=(3, 3), padding=1, stride=1, activation="relu"))
    _(Conv2D(nfilters=8, filter_shape=(3, 3), padding=1, stride=1, activation="relu"))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(128,), activation="relu"))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation="softmax"))

    return model
