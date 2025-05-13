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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from abc import ABC

from .layer import Layer
from pydtnn.utils import encode_tensor

class Input(Layer, ABC):

    #def __init__(self, shape=(1,)):
    #    super().__init__(shape)
#
    #def initialize(self, prev_shape, need_dx=True):
    #    super().initialize(prev_shape, need_dx)
    #    # FIXME 0: If the shape is in format "self.model.tensor_format", this following line will mess up the shape.
    #    self.shape = encode_tensor(self.shape, self.model.tensor_format)

    # Possible FIXME 0 fix:
    def __init__(self, shape:tuple = (1,), is_shape_in_format:bool = False):
        super().__init__(shape)
        self.is_shape_in_format = is_shape_in_format

    def initialize(self, prev_shape:tuple, need_dx:bool=True):
        super().initialize(prev_shape, need_dx)
        if not self.is_shape_in_format:
            self.shape = encode_tensor(self.shape, self.model.tensor_format)