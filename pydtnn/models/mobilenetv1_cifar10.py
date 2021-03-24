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
from ..activations import *


def create_mobilenetv1_cifar10(model):
    first_filters = 32
    _ = model.add
    _( Input(shape=(3, 32, 32)) )
    _( Conv2D(nfilters=first_filters, filter_shape=(3,3), grouping="standard", padding=1, stride=2, activation="relu", use_bias=False))

    layout = [ [64, 1], [128, 2], [256, 2], [512, 6], [1024, 2] ]
    for n_filt, reps in layout:
        for r in range(reps):
            stride = 2 if reps > 1 and r == 0 else 1
            _( Conv2D(nfilters=first_filters, filter_shape=(3, 3), grouping="depthwise", padding=1, stride=stride, use_bias=False) )
            _( BatchNormalization() )
            _( Relu() )
            _( Conv2D(nfilters=n_filt, filter_shape=(1, 1), grouping="pointwise", use_bias=False) )
            _( BatchNormalization() )
            _( Relu() )
            first_filters = n_filt

    _( AveragePool2D(pool_shape=(1,1)) )
    _( Flatten() )
    _( FC(shape=(1024,)) )
    _( FC(shape=(10,), activation="softmax") )
    return model
