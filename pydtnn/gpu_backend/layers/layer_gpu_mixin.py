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


class LayerGPUMixin:
    """
    Mixin used to extend a Layer class with the attributes and methods required
    by GPU Layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = None
        self.y = None
        self.weights_cpu = None
        self.biases_cpu = None
        self.x = None
        self.dx = None
        self.dw = None
        self.db = None
        self.dw_cpu = None
        self.db_cpu = None
        self.one_vec_cpu = None
        self.one_vec_gpu = None

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx)
        self.x = x
