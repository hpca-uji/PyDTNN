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

def switch_losses(loss_func_name: str) -> Loss:
    # From snake to camel, if it's necessary
    _loss_func_name = loss_func_name.split("_")
    if len(_loss_func_name) > 1:
        _loss_func_name = "".join(map(lambda x: x.lower().capitalize(), _loss_func_name))
    else:
        _loss_func_name = loss_func_name

    match _loss_func_name:
        case CategoricalCrossEntropy.__name__:
            return categorical_cross_entropy
        case BinaryCrossEntropy.__name__:
            return binary_cross_entropy
        case _:
            raise NotImplementedError(f"\'{loss_func_name}\' is not implemented!")
