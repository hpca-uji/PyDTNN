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
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.activations import Relu, Softmax

def create_alexnet(input_shape:tuple[int, int, int]=(227, 227, 3), 
                   output_shape:tuple[int, ...] = (1000,)) -> list[LayerAndActivationBase]:
    list_layers:list[LayerAndActivationBase] = list()
    _ = list_layers.append
    _(Input(shape = input_shape))
    _(Conv2D(nfilters=96, filter_shape=(11, 11), padding=0, stride=4, activation=Relu))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))
    _(Conv2D(nfilters=256, filter_shape=(5, 5), padding=2, stride=1, activation=Relu))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))
    _(Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
    _(Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
    _(Conv2D(nfilters=256, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))
    _(Flatten())
    _(FC(shape=(4096,), activation=Relu))
    _(Dropout(rate=0.5))
    _(FC(shape=(4096,), activation=Relu))
    _(Dropout(rate=0.5))
    _(FC(shape = output_shape, activation=Softmax))

    return list_layers
