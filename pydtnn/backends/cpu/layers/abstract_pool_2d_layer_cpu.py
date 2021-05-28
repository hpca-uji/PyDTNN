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

import numpy as np

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import AbstractPool2DLayer
from pydtnn.performance_models import im2col_time, col2im_time


class AbstractPool2DLayerCPU(LayerCPU, AbstractPool2DLayer, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = self.co = self.n = 0

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.hi, self.wi, self.ci = prev_shape
        if self.pool_shape[0] == 0:
            self.pool_shape = (self.hi, self.pool_shape[1])
        if self.pool_shape[1] == 0:
            self.pool_shape = (self.pool_shape[0], self.wi)
        self.kh, self.kw = self.pool_shape
        self.ho = (self.hi + 2 * self.vpadding - self.kh) // self.vstride + 1
        self.wo = (self.wi + 2 * self.hpadding - self.kw) // self.hstride + 1
        assert self.ho > 0 and self.wo > 0
        self.co = self.ci
        self.shape = (self.ho, self.wo, self.co)
        self.n = np.prod(self.shape)

        self.fwd_time = \
            im2col_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            col2im_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) if need_dx else 0
