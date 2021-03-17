"""
PyDTNN Activation layers

If you want to add a new activation layer:
    1) create a new Python file in this directory,
    2) define your activation layer class as derived from Activation (or any Activation derived class),
    3) and, optionally, import your activation layer on this file.
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

from .activation import Activation
from .arctanh import Arctanh
from .log import Log
from .relu import Relu
from .sigmoid import Sigmoid
from .softmax import Softmax
from .tanh import Tanh
from ..utils import get_derived_classes

# Aliases
sigmoid = Sigmoid
relu = Relu
tanh = Tanh
arctanh = Arctanh
log = Log
softmax = Softmax

# Search this module for Activation derived classes and expose them
get_derived_classes(Activation, locals())
