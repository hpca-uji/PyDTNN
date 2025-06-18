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
from ..activations import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from ..layers.conv_2d import GroupingEnum

def create_mobilenetv1(input_shape: Sequence[int], output_shape: Sequence[int]) -> Iterable[LayerAndActivationBase]:
    first_filters = 32
    yield Input(shape=input_shape)
    yield Conv2D(nfilters=first_filters, filter_shape=(3,3), grouping=GroupingEnum.STANDARD, padding=1, stride=2, activation=relu, use_bias=False)

    layout = [ [64, 1], [128, 2], [256, 2], [512, 6], [1024, 2] ]
    for n_filt, reps in layout:
        for r in range(reps):
            stride = 2 if reps > 1 and r == 0 else 1
            yield  Conv2D(nfilters=first_filters, filter_shape=(3, 3), grouping=GroupingEnum.DEPTHWISE, padding=1, stride=stride, use_bias=False) 
            yield  BatchNormalization() 
            yield  Relu() 
            yield  Conv2D(nfilters=n_filt, filter_shape=(1, 1), grouping=GroupingEnum.POINTWISE, use_bias=False) 
            yield  BatchNormalization() 
            yield  Relu() 
            first_filters = n_filt

    yield  AveragePool2D(pool_shape=(1,1)) 
    yield  Flatten() 
    yield  FC(shape=(1024,)) 
    yield  FC(shape=output_shape, activation=softmax) 
