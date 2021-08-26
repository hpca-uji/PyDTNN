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

from abc import ABC

from .conv_2d import Conv2D
from .batch_normalization import BatchNormalization


class Conv2DBatchNormalizationRelu(Conv2D, BatchNormalization, ABC):

    def __init__(self, *args, **kwargs):
        from_parent = kwargs.pop("from_parent", None)
        from_parent2 = kwargs.pop("from_parent2", None)
        if from_parent is None and from_parent2 is None:
            super().__init__(*args, **kwargs)
        else:
            # from_parent.__dict__.pop("forward", None)
            # from_parent2.__dict__.pop("forward", None)
            self.__dict__.update(from_parent.__dict__)
            self.__dict__.update(from_parent2.__dict__)
