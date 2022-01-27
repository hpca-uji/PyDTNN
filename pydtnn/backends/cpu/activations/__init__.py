"""
PyDTNN CPU Activations

If you want to add a new GPU activation layer:
    1) create a new Python file in this directory,
    2) define your CPU activation layer class as derived from ActivationCPU and, optionally, other Activation
       derived class,
    3) and, optionally, import your CPU activation layer on this file.
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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

from pydtnn.utils import get_derived_classes
from .activation_cpu import ActivationCPU
from .arctanh_cpu import ArctanhCPU
from .log_cpu import LogCPU
from .relu_cpu import ReluCPU
from .sigmoid_cpu import SigmoidCPU
from .softmax_cpu import SoftmaxCPU
from .tanh_cpu import TanhCPU

# Search this module for ActivationCPU derived classes and expose them
get_derived_classes(ActivationCPU, locals())
