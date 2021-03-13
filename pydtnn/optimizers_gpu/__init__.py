"""
PyDTNN GPU optimizers

If you want to add a new GPU optimizer:
    1) create a new Python file in this directory,
    2) define your optimizer class as derived from Optimizer (or from a derived class of Optimizer),
    3) and, optionally, import your optimizer on this file.
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

from .adamgpu import AdamGPU
from .nadamgpu import NadamGPU
from .rmspropgpu import RMSPropGPU
from .sgdgpu import SGDGPU

from ..optimizers import Optimizer
from ..utils import get_derived_classes

# Search this module for Optimizer derived classes and expose them
get_derived_classes(Optimizer, locals())

# Aliases
adam_gpu = AdamGPU
nadam_gpu = NadamGPU
rmsprop_gpu = RMSPropGPU
sgd_gpu = SGDGPU
