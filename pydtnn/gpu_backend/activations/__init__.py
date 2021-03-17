"""
PyDTNN GPU Activations

If you want to add a new GPU activation layer:
    1) create a new Python file in this directory,
    2) define your GPU activation layer class as derived from ActivationGPU and Activation (or a derived class),
    3) and, optionally, import your GPU activation layer on this file.
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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

from .activation_gpu import ActivationGPU
from .arctanh_gpu import ArctanhGPU
from .log_gpu import LogGPU
from .relu_gpu import ReluGPU
from .sigmoid_gpu import SigmoidGPU
from .softmax_gpu import SoftmaxGPU
from .tanh_gpu import TanhGPU
from pydtnn.utils import get_derived_classes

# Search this module for ActivationGPU derived classes and expose them
get_derived_classes(ActivationGPU, locals())
