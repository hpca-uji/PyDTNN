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

# noinspection PyUnresolvedReferences
import libcudnn.libcudnn as cudnn
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray

from pydtnn import layers
from .pool_2d_gpu_mixin import Pool2DGPUMixin


class AveragePool2DGPU(Pool2DGPUMixin, layers.AveragePool2D):

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx, x)
        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING']
        self.pool_2d_gpu_initialize(prev_shape, need_dx, x, pool_mode)
