"""
PyDTNN GPU Layers

If you want to add a new GPU layer:
    1) create a new Python file in this directory,
    2) define your layer class as derived from LayerGPU and, optionally, other Layer derived class,
    3) and, optionally, import your GPU layer on this file.
"""

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

from . import memory_allocation

from .layer_gpu import LayerGPU
from .addition_block_gpu import AdditionBlockGPU
from .average_pool_2d_gpu import AveragePool2DGPU
from .batch_normalization_gpu import BatchNormalizationGPU
from .concatenation_block_gpu import ConcatenationBlockGPU
from .conv_2d_gpu import Conv2DGPU
from .dropout_gpu import DropoutGPU
from .fc_gpu import FCGPU
from .flatten_gpu import FlattenGPU
from .input_gpu import InputGPU
from .max_pool_2d_gpu import MaxPool2DGPU
from pydtnn.utils import get_derived_classes

# Search this module for LayerGPU derived classes and expose them
get_derived_classes(LayerGPU, locals())
