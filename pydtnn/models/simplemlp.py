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
from pydtnn.activations import relu, softmax

def create_simplemlp(input_shape: tuple[int, int, int] = (28, 28, 1), 
                     output_shape: tuple[int, ...] = (10,)) -> list[layer.LayerAndActivationBase]:
    list_layers: list[layer.LayerAndActivationBase] = list()
    _ = list_layers.append

    _(Input(shape=input_shape))
    _(Flatten())
    _(FC(shape=(512,), activation=relu))
    _(FC(shape=(512,), activation=relu))
    _(FC(shape=(512,), activation=relu))
    _(FC(shape=output_shape, activation=softmax))

    return list_layers
