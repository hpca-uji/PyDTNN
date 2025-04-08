"""
PyDTNN CPU Layers

If you want to add a new CPU layer:
    1) create a new Python file in this directory,
    2) define your layer class as derived from LayerCPU and, optionally, other Layer derived class,
    3) and, optionally, import your CPU layer on this file.
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

from pydtnn.backends.cpu.layers.layer_cpu import LayerCPU
from pydtnn.utils import get_derived_classes

# Search this module for LayerGPU derived classes and expose them
get_derived_classes(LayerCPU, locals())
