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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import warnings
from abc import ABC

from .layer import Layer
from pydtnn.utils import encode_tensor


class Input(Layer, ABC):
    # NOTE: Input(shape) is expected to be in NHWC

    def __init__(self, shape:tuple = (1,)):
        super().__init__(shape)

    def initialize(self, prev_shape:tuple):
        super().initialize(prev_shape)
