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
from layers.layer import LayerAndActivationBase
from pydtnn.initializers import he_uniform
from pydtnn.activations import relu, softmax

def create_vgg19(input_shape: tuple[int, int, int] = (224, 224, 3), 
                 output_shape: tuple[int, ...] = (1000,)) -> list[LayerAndActivationBase]:
    list_layers: list[LayerAndActivationBase] = list()
    _ = list_layers.append

    _(Input(shape=input_shape))
    conv_pattern = [[2, 64], [2, 128], [4, 256], [4, 512], [4, 512]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=relu,
                     weights_initializer=he_uniform))
        _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(4096,), activation=relu, weights_initializer=he_uniform))
    _(Dropout(rate=0.5))
    _(FC(shape=(4096,), activation=relu, weights_initializer=he_uniform))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation=softmax, weights_initializer=he_uniform))

    return list_layers
