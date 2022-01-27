"""
PyDTNN Layers

If you want to add a new layer:
    1) create a new Python file in this directory,
    2) define your layer class as derived from Layer (or any Layer derived class),
    3) and, optionally, import your layer on this file.
"""

#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
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

from .addition_block import AdditionBlock
from .average_pool_2d import AveragePool2D
from .batch_normalization import BatchNormalization
from .batch_normalization_relu import BatchNormalizationRelu
from .concatenation_block import ConcatenationBlock
from .conv_2d import Conv2D
from .conv_2d_relu import Conv2DRelu
from .conv_2d_batch_normalization import Conv2DBatchNormalization
from .conv_2d_batch_normalization_relu import Conv2DBatchNormalizationRelu
from .dropout import Dropout
from .fc import FC
from .flatten import Flatten
from .input import Input
from .layer import Layer
from .max_pool_2d import MaxPool2D
from .abstract_pool_2d_layer import AbstractPool2DLayer
from ..utils import get_derived_classes

# Search this module for Layer derived classes and expose them
get_derived_classes(Layer, locals())
