#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
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


class ActivationGPU(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = None
        self.y = None
        self.dx = None

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx)
        self.x = x
