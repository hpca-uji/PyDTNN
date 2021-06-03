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
import pycuda.gpuarray as gpuarray

from pydtnn.layers import MaxPool2D
from .abstract_pool_2d_layer_gpu import AbstractPool2DLayerGPU
from ..libs import libcudnn as cudnn


class MaxPool2DGPU(AbstractPool2DLayerGPU, MaxPool2D):

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx, x)
        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_MAX']
        self.initialize_pool_2d_gpu(prev_shape, need_dx, x, pool_mode)
