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

from ..layers import *
from pydtnn.initializers import he_uniform
from pydtnn.activations import relu, softmax

def create_vgg1(input_shape: tuple[int, int, int] = (32, 32, 3), 
                output_shape: tuple[int, ...] = (10,)) -> list[layer.LayerAndActivationBase]:
    list_layers: list[layer.LayerAndActivationBase] = list()
    _ = list_layers.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, activation=relu, weights_initializer=he_uniform))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, activation=relu, weights_initializer=he_uniform))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(128,), activation=relu, weights_initializer=he_uniform))
    _(FC(shape=output_shape, activation=softmax))

    return list_layers
