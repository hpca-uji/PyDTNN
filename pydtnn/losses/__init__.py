"""
Loss classes for classification CNNs

If you want to add a new loss:
    1) create a new Python file in this directory,
    2) define your loss class as derived from Loss (or any Loss derived class),
    3) and, optionally, import your layer on this file.
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

from .binary_cross_entropy import BinaryCrossEntropy
from .categorical_cross_entropy import CategoricalCrossEntropy
from .loss import Loss
from ..utils import get_derived_classes

# Aliases
categorical_cross_entropy = CategoricalCrossEntropy
binary_cross_entropy = BinaryCrossEntropy

# Search this module for Loss derived classes and expose them
get_derived_classes(Loss, locals())
