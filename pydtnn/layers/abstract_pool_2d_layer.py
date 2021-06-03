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

from abc import ABC

from .layer import Layer


class AbstractPool2DLayer(Layer, ABC):

    def __init__(self, pool_shape=(2, 2), padding=0, stride=1):
        super().__init__()
        self.pool_shape = pool_shape
        self.padding = padding
        self.stride = stride
        self.vpadding, self.hpadding = (padding, padding) if isinstance(padding, int) else padding
        self.vstride, self.hstride = (stride, stride) if isinstance(stride, int) else stride

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^24s}|".format(str(self.pool_shape),
                                                f"padd=({self.vpadding},{self.hpadding}), "
                                                f"stride=({self.vstride},{self.hstride})"))
